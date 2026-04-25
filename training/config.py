"""Training configuration: hyperparameters, paths, model settings."""

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_LORA_RANK = 32


def build_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    lora_rank: int = DEFAULT_LORA_RANK,
) -> tuple:
    """Load a 4-bit quantised model with LoRA adapters via Unsloth.

    Returns (model, tokenizer) ready for GRPO training on a single T4.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,          # 4-bit quantisation to fit T4 16GB
        fast_inference=True,         # enable Unsloth fast-inference kernels
        gpu_memory_utilization=0.4,  # reserve 40% VRAM for vLLM inference
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,                              # LoRA rank (higher = more capacity)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,                      # alpha = rank is standard practice
        lora_dropout=0,                            # Unsloth optimised, no dropout needed
        use_gradient_checkpointing="unsloth",      # saves ~30% VRAM
        random_state=42,
    )

    return model, tokenizer


def build_grpo_config(
    output_dir: str = "outputs/grpo",
    **overrides,
) -> "GRPOConfig":
    """Build a GRPOConfig with T4-friendly defaults.

    Any key in *overrides* replaces the corresponding default.
    """
    from trl import GRPOConfig

    defaults = {
        "per_device_train_batch_size": 1,        # T4 16GB fits 1 sample at a time
        "gradient_accumulation_steps": 8,         # effective batch = 1 × 8 = 8
        "num_generations": 4,                     # 4 completions per prompt for GRPO variance
        "learning_rate": 5e-6,                    # conservative LR for LoRA fine-tuning
        "max_prompt_length": 2048,                # system prompt + env state fits in 2K tokens
        "max_completion_length": 1024,            # model output cap per turn
        "num_train_epochs": 1,                    # single pass over dataset
        "use_vllm": True,                         # fast generation via vLLM
        "vllm_mode": "colocate",                  # share GPU between training and vLLM
        "vllm_gpu_memory_utilization": 0.4,       # reserve 40% VRAM for vLLM inference
        "gradient_checkpointing": True,           # trade compute for VRAM
        "gradient_checkpointing_kwargs": {"use_reentrant": False},  # required for LoRA compat
        "beta": 0.04,                             # KL penalty coefficient for GRPO
        "report_to": "trackio",                   # log to trackio dashboard
        "logging_steps": 1,                       # log every step for debugging
        "save_steps": 25,                         # checkpoint every 25 steps
        "output_dir": output_dir,
    }

    defaults.update(overrides)
    return GRPOConfig(**defaults)
