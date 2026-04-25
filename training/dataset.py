"""Dataset construction for GRPO training: task seeds across difficulty tiers."""

from datasets import Dataset


# Constant prompt text — TRL requires a `prompt` column but the actual
# model prompt is built inside the rollout function from env.reset().
_PROMPT = (
    "You are an airline operations agent. A flight has been cancelled. "
    "Rebook all affected passengers onto alternative flights."
)


def build_dataset(
    n_easy: int = 50,
    n_medium: int = 100,
    n_hard: int = 100,
    base_seed: int = 0,
) -> Dataset:
    """Build a HuggingFace Dataset of task seeds for episode sampling.

    Each row carries a seed and tier that the rollout function uses to
    reset the environment deterministically.  The ``prompt`` column is a
    static string satisfying TRL's interface requirement.

    Args:
        n_easy:     Number of easy-tier seeds.
        n_medium:   Number of medium-tier seeds.
        n_hard:     Number of hard-tier seeds.
        base_seed:  Offset added to all seeds for reproducibility.

    Returns:
        HuggingFace Dataset with columns: prompt, seed, tier.
    """
    rows: list[dict] = []
    idx = 0
    for tier, count in [("easy", n_easy), ("medium", n_medium), ("hard", n_hard)]:
        for i in range(count):
            rows.append({
                "prompt": _PROMPT,
                "seed": base_seed + idx,
                "tier": tier,
            })
            idx += 1
    return Dataset.from_list(rows)
