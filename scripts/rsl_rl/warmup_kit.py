"""Populate Kit's extension cache with a single process before multi-GPU training.

Kit downloads some extensions (omni.kit.pip_archive, the URDF importer) on first
use and unpacks them into ~/.local/share/ov/data/exts/v2, which every job shares
through the $HOME mount. When the torchrun ranks unpack concurrently they race,
leaving a half-extracted cache that breaks every later job (e.g. a starlette
without middleware/base.py, which kills `import mlflow`). Running this once per
job, under a flock, makes the ranks find a complete cache.

    python scripts/rsl_rl/warmup_kit.py
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import omni.kit.app  # noqa: E402

# Same pin as isaaclab's UrdfConverter on Isaac Sim 5.1: this version is not in
# the image, so Kit pulls it (and omni.kit.pip_archive) from the registry.
manager = omni.kit.app.get_app().get_extension_manager()
if not manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf-2.4.31", True):
    raise SystemExit("[warmup] failed to enable isaacsim.asset.importer.urdf-2.4.31")

# Both fail if an extension is partially unpacked; for mlflow, Kit's bundled
# starlette shadows the venv's and fastapi imports it.
from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface  # noqa: E402, F401
import mlflow  # noqa: E402, F401

print("[warmup] Kit extension cache OK", flush=True)
simulation_app.close()
