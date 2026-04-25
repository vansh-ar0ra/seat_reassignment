#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# HF Space entrypoint — dispatches jobs via $JOB env var
# ─────────────────────────────────────────────────────────────

echo "══════════════════════════════════════════════════════════"
echo "  JOB          = ${JOB:-idle}"
echo "  PROJECT_REF  = ${PROJECT_REF:-unknown}"
echo "  GPU          = $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
python -c "import trl; print(f'  trl            = {trl.__version__}')" 2>/dev/null || true
python -c "import vllm; print(f'  vllm           = {vllm.__version__}')" 2>/dev/null || true
python -c "from vllm.sampling_params import GuidedDecodingParams; print('  GuidedDecoding = OK')" 2>/dev/null || echo "  GuidedDecoding = MISSING"
echo "══════════════════════════════════════════════════════════"

# ── HF token login ──────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "Logged in to Hugging Face Hub."
else
    echo "WARNING: HF_TOKEN not set — push-to-hub will fail."
fi

# ── Enable hf_transfer for fast uploads ─────────────────────
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# ── Disable FlashInfer JIT (needs nvcc, not in runtime image) ─
# Force TRITON_ATTN backend — FlashInfer needs JIT/nvcc, FLASH_ATTN needs compute >= 8.0
export UNSLOTH_VLLM_NO_FLASHINFER=1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

# ── Start trackio dashboard on port 7860 (default) ──────────
# trackio.show() defaults to port 7860 — avoid passing kwargs
# that may not exist in all trackio versions
python -c "
import trackio, os
os.environ['GRADIO_SERVER_PORT'] = '7860'
os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
trackio.show()
" &
TRACKIO_PID=$!
echo "Trackio dashboard started (PID=$TRACKIO_PID) on :7860"

# ── Job dispatch ────────────────────────────────────────────
case "${JOB:-idle}" in

    idle)
        echo "JOB=idle — container running, trackio dashboard available."
        echo "Set JOB to one of: sft, grpo-smoke, grpo-p1, grpo-p2, eval"
        ;;

    sft)
        echo "Starting SFT training..."
        python training/sft.py || echo "[SFT] exited with code $?"
        ;;

    grpo-smoke)
        echo "Starting GRPO smoke test (2 easy, 2 steps)..."
        python training/train.py --smoke || echo "[GRPO-SMOKE] exited with code $?"
        ;;

    grpo-p1)
        echo "Starting GRPO Phase 1 (50 easy, 200 steps)..."
        python training/train.py --phase 1 || echo "[GRPO-P1] exited with code $?"
        ;;

    grpo-p2)
        echo "Starting GRPO Phase 2 (50 easy + 100 med + 100 hard, 400 steps)..."
        python training/train.py --phase 2 || echo "[GRPO-P2] exited with code $?"
        ;;

    eval)
        echo "Starting evaluation..."
        python training/eval.py --sft --phase1 --phase2 || echo "[EVAL] exited with code $?"
        ;;

    *)
        echo "ERROR: Unknown JOB='${JOB}'. Valid values: idle, sft, grpo-smoke, grpo-p1, grpo-p2, eval"
        ;;

esac

echo "Job '${JOB:-idle}' finished. Keeping container alive for trackio dashboard..."
wait $TRACKIO_PID
