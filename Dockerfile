FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
RUN uv venv --python 3.11
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN uv pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt ./
RUN uv pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py ./

RUN mkdir -p /data/huggingface /data/torch /data/outputs

ENV HOST=0.0.0.0
ENV PORT=8001
ENV IMAGE_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
ENV IMAGE_DEVICE=auto
ENV IMAGE_TORCH_DTYPE=auto
ENV IMAGE_ENABLE_DOCS=true
ENV IMAGE_DEFAULT_STEPS=30
ENV IMAGE_MAX_STEPS=80
ENV IMAGE_MAX_WIDTH=1536
ENV IMAGE_MAX_HEIGHT=1536
ENV IMAGE_OUTPUT_DIR=/data/outputs
ENV IMAGE_OUTPUT_FORMAT=png

ENV HF_HOME=/data/huggingface
ENV HF_HUB_CACHE=/data/huggingface/hub
ENV TRANSFORMERS_CACHE=/data/huggingface/transformers
ENV TORCH_HOME=/data/torch
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "main.py"]
