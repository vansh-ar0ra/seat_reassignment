#!/usr/bin/env bash
set -uo pipefail

# ─────────────────────────────────────────────────────────────
# HF Space entrypoint
#   1. Pull project code from GitHub (clone or update)
#   2. Start trackio dashboard
#   3. Dispatch job via $JOB env var
# ─────────────────────────────────────────────────────────────

# ── HF token login ──────────────────────────────────────────
if [[ -n "${HF_TOKEN:-}" ]]; then
    python3 -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
    echo "Logged in to Hugging Face Hub."
else
    echo "WARNING: HF_TOKEN not set — push-to-hub will fail."
fi

# ── Pull project code ───────────────────────────────────────
PROJECT_DIR=/app/project
GITHUB_REPO="${GITHUB_REPO:-https://github.com/vansh-ar0ra/seat_reassignment.git}"
PROJECT_REF="${PROJECT_REF:-main}"

# For private repos: prefix URL with token
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    GITHUB_REPO_AUTH="${GITHUB_REPO/https:\/\//https:\/\/${GITHUB_TOKEN}@}"
else
    GITHUB_REPO_AUTH="$GITHUB_REPO"
fi

if [[ -d "$PROJECT_DIR/.git" ]]; then
    echo "=== Updating project ($PROJECT_REF) ==="
    cd "$PROJECT_DIR"
    git fetch origin "$PROJECT_REF"
    git reset --hard "origin/$PROJECT_REF"
else
    echo "=== Cloning project ($PROJECT_REF) ==="
    if ! git clone --depth 1 --branch "$PROJECT_REF" "$GITHUB_REPO_AUTH" "$PROJECT_DIR"; then
        echo "FATAL: git clone failed" >&2
        exit 1
    fi
fi

cd "$PROJECT_DIR" || { echo "FATAL: cannot cd to $PROJECT_DIR" >&2; exit 1; }
echo "=== Project at $(git rev-parse --short HEAD) ==="

# Install project-local deps if present
# --no-deps: all deps already installed at build time; just make the project importable
[[ -f setup.py ]] && pip install --no-deps -e . --quiet
[[ -f pyproject.toml ]] && pip install --no-deps -e . --quiet

# ── Runtime env vars ─────────────────────────────────────────
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# Let vllm auto-select attention backend (FLASH_ATTN on A100)

# Make CUDA errors synchronous for accurate tracebacks
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# ── Banner ───────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════"
echo "  JOB          = ${JOB:-idle}"
echo "  PROJECT_REF  = ${PROJECT_REF}"
echo "  COMMIT       = $(git rev-parse --short HEAD)"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
python3 -c "import trl; print(f'  trl            = {trl.__version__}')" 2>/dev/null || true
python3 -c "import vllm; print(f'  vllm           = {vllm.__version__}')" 2>/dev/null || true
echo "══════════════════════════════════════════════════════════"

# ── Wait for env Space to be reachable ──────────────────────
if [[ -n "${ENV_URL:-}" ]] && [[ "${JOB:-}" =~ ^(grpo-smoke|grpo-p1|grpo-p2|eval)$ ]]; then
    echo "Pinging env Space at $ENV_URL ..."
    for i in {1..30}; do
        if curl -fsS "${ENV_URL}/health" > /dev/null 2>&1; then
            echo "Env Space healthy."
            break
        fi
        echo "  attempt $i/30: env not ready, sleeping 10s..."
        sleep 10
    done
fi

# ── Start trackio dashboard on port 7860 ─────────────────────
mkdir -p "${TRACKIO_DIR:-/data/trackio}"
python3 -c "
import trackio, os
os.environ['GRADIO_SERVER_PORT'] = '7860'
os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
trackio.show()
" &
TRACKIO_PID=$!
echo "Trackio dashboard started (PID=$TRACKIO_PID) on :7860"

# ── Job dispatch ─────────────────────────────────────────────
case "${JOB:-idle}" in

    idle)
        echo "JOB=idle — container running, trackio dashboard available."
        echo "Set JOB to one of: sft, grpo-smoke, grpo-p1, grpo-p2, eval"
        ;;

    sft)
        echo "Starting SFT training..."
        python3 training/sft.py || echo "[SFT] exited with code $?" >&2
        ;;

    grpo-smoke)
        echo "Starting GRPO smoke test (2 easy, 2 steps)..."
        python3 training/train.py --smoke || echo "[GRPO-SMOKE] exited with code $?" >&2
        ;;

    grpo-p1)
        echo "Starting GRPO Phase 1 (50 easy, 200 steps)..."
        python3 training/train.py --phase 1 || echo "[GRPO-P1] exited with code $?" >&2
        ;;

    grpo-p2)
        echo "Starting GRPO Phase 2 (50 easy + 100 med + 100 hard, 400 steps)..."
        python3 training/train.py --phase 2 || echo "[GRPO-P2] exited with code $?" >&2
        ;;

    eval)
        echo "Starting evaluation..."
        python3 training/eval.py --sft --phase1 --phase2 || echo "[EVAL] exited with code $?" >&2
        ;;

    *)
        echo "ERROR: Unknown JOB='${JOB}'. Valid values: idle, sft, grpo-smoke, grpo-p1, grpo-p2, eval" >&2
        ;;

esac

echo "=== Job '${JOB:-idle}' finished; trackio still serving ==="
wait $TRACKIO_PID
