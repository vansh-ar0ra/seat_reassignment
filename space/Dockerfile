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

FROM vllm/vllm-openai:v0.11.0
# Strip the default entrypoint — we use our own
ENTRYPOINT []

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRACKIO_DIR=/data/trackio

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: deps — rebuilt only when requirements-train.txt changes
COPY requirements-train.txt /app/requirements-train.txt
RUN pip install --no-cache-dir --no-build-isolation -r /app/requirements-train.txt

# Build-time verification — fail fast if pins are broken
RUN python3 -c "import trl; print('trl', trl.__version__)" && \
    python3 -c "import vllm; print('vllm', vllm.__version__)"

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
