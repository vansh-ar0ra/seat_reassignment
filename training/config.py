"""Training configuration: hyperparameters, paths, model settings."""

import torch

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_SEQ_LENGTH = 4096
DEFAULT_LORA_RANK = 32


def build_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    lora_rank: int = DEFAULT_LORA_RANK,
) -> tuple:
    """Load a 4-bit quantised model with LoRA adapters via plain HF + PEFT.

    Returns (model, tokenizer) ready for GRPO training on A100 (80GB VRAM).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def build_grpo_config(
    output_dir: str = "outputs/grpo",
    **overrides,
) -> "GRPOConfig":
    """Build a GRPOConfig with A100-friendly defaults (80GB VRAM).

    Any key in *overrides* replaces the corresponding default.
    """
    from trl import GRPOConfig

    defaults = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,         # effective batch = 4 × 8 = 32
        "num_generations": 4,                     # 4 completions per prompt for GRPO variance
        "learning_rate": 2e-6,                    # conservative LR for LoRA fine-tuning
        "max_prompt_length": 2048,                # system prompt + env state fits in 2K tokens
        "max_completion_length": 1024,            # model output cap per turn
        "num_train_epochs": 1,                    # single pass over dataset
        "use_vllm": True,                         # fast generation via vLLM
        "vllm_mode": "colocate",                  # share GPU between training and vLLM
        "vllm_gpu_memory_utilization": 0.3,       # single vLLM, keep headroom for grads + optimizer
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
