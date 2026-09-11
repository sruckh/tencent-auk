# Tencent AuK RunPod serverless worker — stage 05 of the ICM pipeline (.icm/).
# Layering contract: .icm/stages/05-container-and-dockerfile/output/container-and-dockerfile.md
FROM nvidia/cuda:12.8.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System + audio codecs (libsndfile for soundfile, ffmpeg/sox for ingest
# conversion), installed lean in one layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        ffmpeg libsndfile1 sox \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Virtualenv: keep system Python pristine.
RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch cu128 (CUDA 12.8 match). Deployment decision 2026-09-11 (rev. 2):
# the 2.8 line deliberately sits outside upstream's declared 2.7 range —
# see stage 05 spec.
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

# Worker-level pins first: this layer caches across worker-code changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Upstream core library. torch 2.8 is outside upstream's declared 2.7 range
# (recorded deployment decision, rev. 2), so install --no-deps — its remaining
# inference deps are pinned in requirements.txt, and `pip check` is omitted
# deliberately. No flash-attention wheel: the inference path pins
# attn_backend="torch" (infer_auk.py). The import smoke runs in the image
# build only, never on the authoring host.
COPY src/ /app/src/
RUN pip install --no-deps /app/src \
    && python3 -c "from auk.infer.infer_auk import AukInfer"

# Worker code last — changes here rebuild only this layer.
COPY schema_validator.py engine.py storage.py handler.py /app/

# Pinned environment (runpod-invariants.md / s3-storage.md). Credentials are
# NOT baked in — AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY arrive as RunPod
# endpoint env vars at runtime.
ENV RUNPOD_LOG_LEVEL=INFO \
    RUNPOD_INIT_TIMEOUT=1200 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    AUK_OFFLINE=0 \
    DEFAULT_MODEL_VARIANT=flash \
    CKPT_ROOT=/runpod-volume/ckpts \
    S3_ENDPOINT_URL=https://s3.us-west-004.backblazeb2.com \
    S3_BUCKET=auk-audio-production \
    S3_REGION=us-west-004 \
    S3_KEY_PREFIX=auk/ \
    S3_PRESIGN_EXPIRY_SECONDS=86400 \
    S3_ADDRESSING_STYLE=path \
    AUDIO_DELIVERY=auto

# Native cache is populated by RunPod; missing components may use CKPT_ROOT.
RUN mkdir -p /runpod-volume/ckpts /runpod-volume/huggingface-cache/hub
VOLUME /runpod-volume

# Launch contract: unbuffered handler; runpod.serverless.start inside the module.
CMD ["python3", "-u", "handler.py"]
