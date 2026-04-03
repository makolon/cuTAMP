FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG CUTAMP_PATH_ARG=/workspace
ENV CUTAMP_PATH=${CUTAMP_PATH_ARG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/root/.local/bin:$PATH \
    XDG_RUNTIME_DIR=/tmp/runtime-root \
    OMNI_KIT_ALLOW_ROOT=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf autoconf-archive automake bison flex gperf m4 meson ninja-build \
    build-essential pkg-config \
    git curl unzip zip tar \
    python3 python3-dev python3-pip python3-venv \
    cmake ffmpeg \
    \
    # X11 / EGL / Vulkan tools (GUI debug + headless)
    xauth x11-utils \
    libdbus-1-3 libglib2.0-0 libsm6 libfontconfig1 libfreetype6 \
    libx11-6 libx11-dev libxext6 libxext-dev libxrender1 libxrender-dev \
    libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render0 \
    libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-xfixes0 libxcb-xinerama0 libxcb-xkb1 \
    libxkbcommon0 libxkbcommon-x11-0 libxkbfile1 libxmu6 libxaw7 libxxf86dga1 \
    libgl1 libgl1-mesa-dev libopengl0 libglu1-mesa-dev freeglut3-dev \
    libegl1 libglew-dev libgl1-mesa-dev libglu1-mesa-dev \
    vulkan-tools zenity \
    \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR ${CUTAMP_PATH}

CMD ["/bin/bash"]
