# deep-robotics-mimic training image for the L3 cluster (Slurm + pyxis/enroot).
#
# Only the dependency closure is baked in. The repo is bind-mounted at
# /workspace by the job and imported through PYTHONPATH, so pulling a commit or
# adding motions does not need a rebuild.
#
# Ubuntu 24.04 rather than 22.04: Isaac Sim 5.1 needs CXXABI_1.3.15 (GCC 13+),
# which 22.04's libstdc++ lacks -- the same problem scripts/setup_conda_runtime.py
# patches with an LD_PRELOAD on the conda side.
#
# On the cluster, build with docker/build_cluster_image.sh instead; it runs the
# same install_deps.sh under pyxis and needs no docker daemon.
#
# Local build (context is this directory only; nothing from the repo is copied):
#   docker build -t deep-robotics-mimic:latest docker/
#
# Smoke test on any GPU box:
#   docker run --rm --gpus all -v $PWD:/workspace deep-robotics-mimic:latest \
#     python scripts/rsl_rl/train.py --task=Tracking-Flat-DR02_PRO \
#       --registry_name dataset/gmr/<motion>.npz --headless --num_envs 512 --max_iterations 5

FROM ubuntu:24.04

# All install steps live in install_deps.sh so the cluster build
# (build_cluster_image.sh) produces the same filesystem. One RUN on purpose: the
# chmod at its end would duplicate ~20 GB if it landed in a separate layer.
COPY install_deps.sh /tmp/install_deps.sh
RUN --mount=type=cache,target=/root/.cache/pip bash /tmp/install_deps.sh && rm /tmp/install_deps.sh

# Runtime environment. Keep in sync with image.env, which carries the same
# values into the pyxis-built image (enroot reads /etc/environment, not ENV).
ENV PATH=/opt/venv/bin:$PATH
ENV VIRTUAL_ENV=/opt/venv
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
# Read by both the docker and enroot NVIDIA hooks; graphics is needed for Vulkan.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PYTHONPATH=/workspace/source/whole_body_tracking
ENV PYTHONUNBUFFERED=1

# Let git read the bind-mounted repo whatever uid the container runs as.
ENV GIT_CONFIG_COUNT=1
ENV GIT_CONFIG_KEY_0=safe.directory
ENV GIT_CONFIG_VALUE_0=*

WORKDIR /workspace
CMD ["python", "-c", "import torch, isaaclab, rsl_rl; print('deep-robotics-mimic image OK', torch.__version__, torch.cuda.get_arch_list())"]
