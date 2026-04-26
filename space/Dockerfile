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
        git curl gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Cache-bust: Docker invalidates this layer when the branch HEAD changes
ADD https://api.github.com/repos/vansh-ar0ra/seat_reassignment/git/refs/heads/single-step /tmp/.git-ref

# Clone project at build time
RUN git clone --branch ${PROJECT_REF} --depth 1 ${PROJECT_REPO} /app

WORKDIR /app

# Single resolution pass — all version pins in requirements-train.txt
RUN pip install --no-cache-dir --no-build-isolation -r requirements-train.txt

# Build-time verification of the imports that actually fail at runtime
RUN python -c "import trl; print('trl', trl.__version__)" && \
    python -c "import vllm; print('vllm', vllm.__version__)" && \
    python -c "import unsloth_zoo; print('unsloth_zoo', unsloth_zoo.__version__)" && \
    python -c "from unsloth import FastLanguageModel; print('unsloth import OK')" && \
    python -c "from unsloth_zoo.vllm_lora_request import LoRARequest; print('unsloth_zoo+vllm OK')" && \
    python -c "from vllm.sampling_params import GuidedDecodingParams; print('GuidedDecodingParams OK')"

# Trackio dashboard port
EXPOSE 7860

# Copy entrypoint (lives in space/ but we need it at /app root for the image)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
