#!/bin/bash
# Single-motion DR02_pro tracking training on Slurm, inside the pyxis image
# built from docker/Dockerfile.
#
# USAGE (submit from anywhere inside the deep-robotics-mimic checkout):
#   sbatch -J boxing scripts/slurm/train_DR02_pro.sbatch dataset/gmr/boxing.npz
#   sbatch -J bow    scripts/slurm/train_DR02_pro.sbatch dataset/gmr/bow.npz --seed 7
#
#   Arg 1 is the FK npz, relative to the repo root; everything after it is
#   forwarded to scripts/rsl_rl/train.py unchanged.
#
# Environment knobs (all optional, pass with `sbatch --export=ALL,KNOB=...`):
#   NUM_ENVS=16384         parallel envs PER GPU (train.py's --num_envs is per process,
#                          despite the README); 65,536 in total on the default 4 GPUs.
#   MAX_ITER=30000         PPO iterations; the task default of 100k is overkill for one clip
#   RUN_NAME=<motion>      suffix of logs/rsl_rl/DR02_pro_flat/<timestamp>_<RUN_NAME>
#   GPUS=<allocated>       ranks for torch.distributed; defaults to the GPUs Slurm allocated,
#                          so change the count with `sbatch --gres=gpu:N`, not here.
#   LOGGER=tensorboard     tensorboard | wandb | neptune
#   RESUME_RUN=<dir>       folder name under logs/rsl_rl/DR02_pro_flat/ to resume from
#   RESUME_CKPT=<file>     checkpoint filename in that folder, e.g. model_12000.pt
#   IMAGE=<sqsh>           defaults to /tier1/wbc/images/deep-robotics-mimic.sqsh,
#                          built by docker/build_cluster_image.sh
#
# Checkpoints are written only every save_interval (2000 in rsl_rl_ppo_cfg.py) and
# the trainer does not checkpoint on SIGTERM, so a job that hits --time loses up
# to that many iterations. Size --time generously, or lower save_interval.

#SBATCH --job-name=dr02_mimic
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=raus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00

set -uo pipefail

MOTION=${1:?usage: sbatch $0 <motion.npz> [extra train.py args...]}
shift

# Slurm runs a spooled copy of this file, so locate the repo from the submit dir.
REPO=${REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null)}
: "${REPO:?submit from inside the deep-robotics-mimic checkout, or set REPO}"

IMAGE=${IMAGE:-/tier1/wbc/images/deep-robotics-mimic.sqsh}
NUM_ENVS=${NUM_ENVS:-16384}
MAX_ITER=${MAX_ITER:-30000}
RUN_NAME=${RUN_NAME:-$(basename "$MOTION" .npz)}
GPUS=${GPUS:-${SLURM_GPUS_ON_NODE:-1}}
LOGGER=${LOGGER:-tensorboard}

TRAIN_ARGS=(
    --task=Tracking-Flat-DR02_PRO
    --registry_name "$MOTION"
    --logger "$LOGGER"
    --run_name "$RUN_NAME"
    --headless
    --num_envs "$NUM_ENVS"
    --max_iterations "$MAX_ITER"
)
if [ "$GPUS" -gt 1 ]; then
    TRAIN_ARGS+=(--distributed)
else
    TRAIN_ARGS+=(--device cuda:0)
fi
if [ -n "${RESUME_RUN:-}" ]; then
    TRAIN_ARGS+=(--resume True --load_run "$RESUME_RUN" --checkpoint "${RESUME_CKPT:?set RESUME_CKPT with RESUME_RUN}")
fi
TRAIN_ARGS+=("$@")

echo "[job] node=$(hostname) job=${SLURM_JOB_ID:-?} repo=$REPO image=$IMAGE"
echo "[job] motion=$MOTION envs=$NUM_ENVS/GPU x $GPUS GPUs = $((NUM_ENVS * GPUS)) iters=$MAX_ITER gpus=$GPUS run=$RUN_NAME"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/[job]   /'

# train.py writes logs/ relative to the cwd, so they land in $REPO/logs.
# $HOME is mounted so Kit's shader/extension caches persist between jobs.
# --container-writable: Kit writes caches and logs inside its package tree.
srun --ntasks=1 --nodes=1 \
     --container-image="$IMAGE" \
     --container-mounts="$REPO:/workspace,$HOME:$HOME" \
     --container-workdir=/workspace \
     --container-writable \
     python scripts/rsl_rl/train_auto_restart.py --nproc_per_node "$GPUS" "${TRAIN_ARGS[@]}"
