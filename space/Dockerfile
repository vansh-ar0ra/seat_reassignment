# ─────────────────────────────────────────────────────────────
# Training container for HuggingFace Space (T4 medium)
# ─────────────────────────────────────────────────────────────
#
# Build args:
#   PROJECT_REPO  — public GitHub URL (default: team repo)
#   PROJECT_REF   — branch / tag / SHA to checkout (default: single-step)
#
# For a PRIVATE repo, replace the git-clone block with:
#   RUN --mount=type=secret,id=GITHUB_TOKEN \
#       git clone https://$(cat /run/secrets/GITHUB_TOKEN)@github.com/vansh-ar0ra/seat_reassignment.git /app \
#       && cd /app && git checkout ${PROJECT_REF}
# ─────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ARG PROJECT_REPO=https://github.com/vansh-ar0ra/seat_reassignment.git
ARG PROJECT_REF=single-step

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl && \
    rm -rf /var/lib/apt/lists/*

# Clone project at build time
RUN git clone ${PROJECT_REPO} /app && \
    cd /app && git checkout ${PROJECT_REF}

WORKDIR /app

# Install Python dependencies
# 1. Install unsloth first (it pulls its own vllm/trl — often too new)
# 2. Then force-install pinned trl + vllm from requirements-train.txt
# 3. Then extras
RUN pip install --no-cache-dir unsloth && \
    pip install --no-cache-dir -r requirements-train.txt && \
    pip install --no-cache-dir --force-reinstall "trl>=0.17.0,<0.20.0" "vllm>=0.6.0,<0.12.0" && \
    pip install --no-cache-dir trackio huggingface_hub hf_transfer

# Log installed versions for debugging
RUN python -c "import trl; print('trl', trl.__version__)" && \
    python -c "import vllm; print('vllm', vllm.__version__)" && \
    python -c "from vllm.sampling_params import GuidedDecodingParams; print('GuidedDecodingParams OK')"

# Trackio dashboard port
EXPOSE 7860

# Copy entrypoint (lives in space/ but we need it at /app root for the image)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
