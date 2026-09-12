FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and essential build tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    ninja-build \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up a working directory inside the container
WORKDIR /workspace

# Tell uv it is safe to break the system package rule inside this container
ENV UV_BREAK_SYSTEM_PACKAGES=1

# 2. First layer: Install bleeding-edge AI libraries for Blackwell (sm_120)
RUN uv pip install --system --pre torch torchvision torchaudio --torch-backend=cu130

# 2. Second layer: Deep Ecosystem & ComfyUI Core packages
# Using precise constraints from your ecosystem specifications
RUN uv pip install --system \
    "diffusers" \
    "einops" \
    "huggingface_hub>=0.25.2" \
    "kornia" \
    "ninja~=1.11.1" \
    "transformers[timm]>=4.50.0" \
    "accelerate" \
    "sentencepiece" \
    "fastapi" \
    "faster-whisper" \
    "yt-dlp" \
    "uvicorn" \
    "nvidia-cublas-cu12" \
    "nvidia-cudnn-cu12"

# 5. Link the global python site-packages paths so faster-whisper can find its .so.12 files
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

#6. deno runtime
# 1. Install unzip safely down here to support the deno installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Grab Deno using the modern official script
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh

EXPOSE 8880 8881 8882 8883 8884 8885

# Keep container alive if started without a default command
CMD ["sleep", "infinity"]

