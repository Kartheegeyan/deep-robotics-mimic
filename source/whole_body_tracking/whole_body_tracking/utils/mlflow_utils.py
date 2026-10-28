"""MLflow logging backend for rsl_rl 5.x.

rsl_rl's Logger only knows tensorboard, wandb and neptune. MlflowLogger adds
``logger="mlflow"``: TensorBoard event files are still written to the log dir,
and every scalar, the run config, the git diff and the checkpoints also go to
the MLflow tracking server.

Configured through the environment:
    MLFLOW_TRACKING_URI      e.g. http://10.83.0.10:5001 (required)
    MLFLOW_EXPERIMENT_NAME   defaults to the runner's experiment_name
"""

from __future__ import annotations

import os
import re
import socket

import git
import mlflow
from rsl_rl.utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter

# MLflow accepts alphanumerics, underscores, dashes, periods, spaces and slashes in keys.
_INVALID_KEY_CHARS = re.compile(r"[^\w\-. /]")


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[_INVALID_KEY_CHARS.sub("_", key)] = str(v)[:500]
    return out


def _git_tags(path: str) -> dict[str, str]:
    try:
        repo = git.Repo(path, search_parent_directories=True)
    except Exception:
        print(f"[MLflow] No git repository found at {path}; run is not tagged with a commit.")
        return {}
    tags = {
        "mlflow.source.git.commit": repo.head.commit.hexsha,
        "git_commit": repo.head.commit.hexsha,
        "git_dirty": str(repo.is_dirty(untracked_files=False)),
    }
    try:
        tags["git_branch"] = repo.active_branch.name
    except TypeError:  # detached HEAD
        pass
    return tags


class MlflowSummaryWriter(SummaryWriter):
    """TensorBoard writer that also sends every scalar to an MLflow run."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict, tags: dict[str, str]) -> None:
        super().__init__(log_dir, flush_secs=flush_secs)
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            raise KeyError("Set MLFLOW_TRACKING_URI (e.g. http://10.83.0.10:5001) to log to MLflow.")

        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME") or cfg["experiment_name"])
        self.run = mlflow.start_run(run_name=os.path.basename(log_dir), tags=tags)
        print(f"[MLflow] Logging to run '{self.run.info.run_name}' ({self.run.info.run_id})")
        # Scalars are batched per step and sent asynchronously so the server
        # round trip stays off the training loop.
        self._pending: dict[int, dict[str, float]] = {}

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        mlflow.log_params(_flatten(train_cfg), synchronous=False)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False) -> None:
        super().add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime, new_style=new_style)
        step = int(global_step or 0)
        self._pending.setdefault(step, {})[_INVALID_KEY_CHARS.sub("_", tag)] = float(scalar_value)

    def flush_metrics(self) -> None:
        for step, metrics in self._pending.items():
            mlflow.log_metrics(metrics, step=step, synchronous=False)
        self._pending.clear()

    def save_model(self, model_path: str, it: int) -> None:
        mlflow.log_artifact(model_path, artifact_path="checkpoints")

    def save_file(self, path: str, artifact_path: str = "git") -> None:
        mlflow.log_artifact(path, artifact_path=artifact_path)

    def stop(self) -> None:
        self.flush_metrics()
        self.close()
        mlflow.end_run()


class MlflowLogger(Logger):
    """rsl_rl Logger with an extra "mlflow" backend.

    Set ``registry_name`` (the motion file) before training so it is tagged on the run.
    """

    registry_name: str | None = None

    def init_logging_writer(self) -> None:
        if self.cfg.get("logger", "").lower() != "mlflow":
            return super().init_logging_writer()

        self.logger_type = "mlflow"
        self.writer = None
        if self.log_dir is not None and not self.disable_logs:
            tags = _git_tags(os.path.dirname(os.path.abspath(__file__)))
            tags.update({
                "motion": str(self.registry_name),
                "log_dir": self.log_dir,
                "host": socket.gethostname(),
                "num_gpus": str(self.gpu_world_size),
            })
            if os.environ.get("SLURM_JOB_ID"):
                tags["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
            if self.cfg.get("resume"):
                tags["resumed_from"] = f"{self.cfg.get('load_run')}/{self.cfg.get('load_checkpoint')}"
            self.writer = MlflowSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg, tags=tags)

        files_to_upload = self._store_code_state()
        if self.writer is not None:
            self.writer.store_config(self.env_cfg, self.cfg)
            for path in files_to_upload:
                self.writer.save_file(path)
            # train.py dumps env.yaml / agent.yaml here before learn() starts.
            params_dir = os.path.join(self.log_dir, "params")
            if os.path.isdir(params_dir):
                mlflow.log_artifacts(params_dir, artifact_path="params")

    def log(self, *args, **kwargs) -> None:
        super().log(*args, **kwargs)
        if self.writer is not None and self.logger_type == "mlflow":
            self.writer.flush_metrics()

    def save_model(self, path: str, it: int) -> None:
        if self.writer is not None and self.logger_type == "mlflow":
            self.writer.save_model(path, it)
        else:
            super().save_model(path, it)

    def stop_logging_writer(self) -> None:
        if self.writer is not None and self.logger_type == "mlflow":
            self.writer.stop()
        else:
            super().stop_logging_writer()
