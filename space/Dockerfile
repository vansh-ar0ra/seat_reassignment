# ─────────────────────────────────────────────────────────────
# Training container for HuggingFace Space (A100 large)
# ─────────────────────────────────────────────────────────────
#
# Image = deps + entrypoint only.  Project code is pulled at
# container start by entrypoint.sh → restart ≈ 30 s vs rebuild ≈ 8 min.
#
# Rebuild only when:
#   • requirements-train.txt changes
#   • Dockerfile or entrypoint.sh changes
# ─────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRACKIO_DIR=/data/trackio

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: deps — rebuilt only when requirements-train.txt changes
COPY requirements-train.txt /app/requirements-train.txt
RUN pip install --no-cache-dir --no-build-isolation -r /app/requirements-train.txt

# Build-time verification — fail fast if pins are broken
# (unsloth/unsloth_zoo need a GPU at import time, so only check vllm/trl here)
RUN python -c "import trl; print('trl', trl.__version__)" && \
    python -c "import vllm; print('vllm', vllm.__version__)" && \
    python -c "from vllm.adapter_commons.request import AdapterRequest; print('vllm.adapter_commons OK')" && \
    python -c "from vllm.sampling_params import GuidedDecodingParams; print('GuidedDecodingParams OK')"

# Layer 2: entrypoint — code is NOT in the image, pulled at runtime
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# HF Spaces runs containers as uid 1000.  Pre-create all dirs it needs.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/project /data/trackio /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /data /home/appuser
USER appuser
ENV HOME=/home/appuser

EXPOSE 7860
CMD ["/app/entrypoint.sh"]
