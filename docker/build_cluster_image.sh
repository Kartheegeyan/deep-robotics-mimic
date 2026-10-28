#!/bin/bash
# Builds the deep-robotics-mimic training image ON THE CLUSTER, no docker daemon
# needed. Run it on the login node from inside the checkout:
#
#   bash docker/build_cluster_image.sh
#
# It submits one Slurm job that starts a plain ubuntu:24.04 container under
# pyxis as (remapped) root, runs docker/install_deps.sh -- the same script the
# Dockerfile runs -- installs docker/image.env as /etc/environment (enroot's
# equivalent of ENV), and saves the result with --container-save. A second step
# opens the new image with a GPU and checks CUDA before it replaces $OUT.
#
# Knobs (all optional):
#   OUT=/tier1/wbc/images/deep-robotics-mimic.sqsh   where the image lands
#   PARTITION=raus
#   BASE=ubuntu:24.04                  base image, pulled by pyxis/enroot
#   PIP_CACHE=$HOME/.cache/pip         persisted across builds, kept OUT of the image
#   LOG_DIR=<repo>/logs/slurm
#
# The compute node needs outbound access to Docker Hub, the Ubuntu archive and
# deadsnakes PPA, PyPI, download.pytorch.org and pypi.nvidia.com.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)

OUT=${OUT:-/tier1/wbc/images/deep-robotics-mimic.sqsh}
PARTITION=${PARTITION:-raus}
BASE=${BASE:-ubuntu:24.04}
PIP_CACHE=${PIP_CACHE:-$HOME/.cache/pip}
LOG_DIR=${LOG_DIR:-$REPO/logs/slurm}

mkdir -p "$LOG_DIR" "$PIP_CACHE" "$(dirname "$OUT")"

# Everything not escaped is expanded now, at submit time; \$ is expanded in the job.
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=build_dr_mimic_image
#SBATCH --output=$LOG_DIR/%x-%j.out
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -euo pipefail
TMP="$OUT.building-\$SLURM_JOB_ID"
trap 'rm -f "\$TMP"' EXIT
echo "[build] node=\$(hostname) base=$BASE -> $OUT"

# Mounts are not part of the saved filesystem, so the pip cache stays outside it.
srun --ntasks=1 \\
     --container-image="$BASE" \\
     --container-remap-root \\
     --container-writable \\
     --container-save="\$TMP" \\
     --container-mounts="$HERE:/build:ro,$PIP_CACHE:/pipcache" \\
     bash -c 'PIP_CACHE_DIR=/pipcache bash /build/install_deps.sh && cp /build/image.env /etc/environment'

# Open the new image the way training jobs will -- non-root, GPU attached -- and
# fail before replacing the old image if CUDA or the packages are not usable.
srun --ntasks=1 \\
     --container-image="\$TMP" \\
     python -c "import torch, isaaclab, rsl_rl, importlib.metadata as m; \\
assert torch.cuda.is_available(), 'CUDA not visible in the image'; \\
print('[build] torch', torch.__version__, 'on', torch.cuda.get_device_name(0)); \\
print('[build] isaaclab', m.version('isaaclab'), 'isaacsim', m.version('isaacsim'), 'rsl-rl-lib', m.version('rsl-rl-lib'))"

mv -f "\$TMP" "$OUT"
trap - EXIT
echo "[build] done: $OUT (\$(du -h "$OUT" | cut -f1))"
EOF

echo "Build log: $LOG_DIR/build_dr_mimic_image-<jobid>.out"
