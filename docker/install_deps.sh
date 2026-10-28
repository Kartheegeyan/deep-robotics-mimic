#!/bin/bash
# Installs the deep-robotics-mimic dependency closure into an Ubuntu 24.04 root
# filesystem. The single source of truth for both image builds:
#   docker/Dockerfile               (docker build, then enroot import)
#   docker/build_cluster_image.sh   (pyxis --container-save, on the cluster)
#
# Must run as root (or under enroot --container-remap-root).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
# Rootless enroot cannot setgroups to apt's _apt sandbox user.
APT=(apt-get -o APT::Sandbox::User=root)

# Isaac Sim is headless here but Kit still loads Vulkan/EGL/X client libraries.
# No Vulkan/EGL manifests are written: with graphics capability the NVIDIA hook
# (docker and enroot both use libnvidia-container) mounts
# /etc/vulkan/icd.d/nvidia_icd.json itself, and a second copy makes Kit report
# "Multiple Installable Client Drivers ... for the same GPU".
"${APT[@]}" update
"${APT[@]}" install -y --no-install-recommends \
    software-properties-common gpg-agent ca-certificates curl git
add-apt-repository -y ppa:deadsnakes/ppa
"${APT[@]}" update
"${APT[@]}" install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    build-essential \
    libgl1 libegl1 libglu1-mesa libvulkan1 libgomp1 libglib2.0-0 \
    libxrandr2 libxinerama1 libxcursor1 libxi6 libxt6 libsm6 libice6 libxkbcommon0
rm -rf /var/lib/apt/lists/*

python3.11 -m venv /opt/venv
export PATH=/opt/venv/bin:$PATH
export PIP_DEFAULT_TIMEOUT=180

# Same order as the README, except torch comes from the cu128 index: the PyPI
# 2.7.0 wheel is cu126 and has no sm_120 kernels for RTX 6000 Pro / 50-series.
# The last install restates isaacsim-kernel's exact pins for click,
# typing_extensions and websockets; left free, wandb drags all three forward.
pip install --upgrade pip
pip install setuptools==80.9.0 wheel packaging
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install flatdict==4.0.1 --no-build-isolation
pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com --no-build-isolation
pip install rsl-rl-lib==5.0.1
pip install psutil onnx onnxscript "wandb>=0.19" toml tensorboard mujoco \
    click==8.1.7 typing_extensions==4.12.2 websockets==12.0

# enroot runs the job as the submitting uid, and Kit writes caches and logs
# inside its own package tree, which would otherwise be root-owned.
chmod -R a+rwX /opt/venv

# Builds have no GPU, so get_arch_list() is empty here; the CUDA tag is the
# build-time proxy for sm_120 support.
python -c "import torch; assert torch.version.cuda == '12.8', torch.version.cuda"
