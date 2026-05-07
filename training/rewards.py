"""Training-side reward shaping and advantage computation.

Weights are applied inside each function so the GRPOTrainer sums them
directly (no ``reward_weights`` kwarg needed).

TRL may forward fewer items in kwargs than len(completions) due to
generation_batch_size slicing. Each function broadcasts raw values to
match len(completions) when possible.
"""


def _broadcast(raw: list, n: int) -> list[float]:
    """Broadcast raw rewards to length n.

    If raw has the right length, use as-is. If n is a whole multiple of
    len(raw), repeat each value. Otherwise fall back to zeros.
    """
    if len(raw) == n:
        return list(raw)
    if len(raw) > 0 and n % len(raw) == 0:
        repeats = n // len(raw)
        return [v for v in raw for _ in range(repeats)]
    return [0.0] * n


def reward_coverage(completions: list, **kwargs) -> list[float]:
    n = len(completions)
    raw = kwargs.get("coverage_reward", [])
    values = _broadcast(raw, n)
    result = [v * 0.35 for v in values]
    print(f"[REWARD coverage] completions={n}, raw={len(raw)}, returning={len(result)}: {result}")
    return result


def reward_ssr_integrity(completions: list, **kwargs) -> list[float]:
    n = len(completions)
    raw = kwargs.get("ssr_integrity_reward", [])
    values = _broadcast(raw, n)
    result = [v * 0.20 for v in values]
    print(f"[REWARD ssr_integrity] completions={n}, raw={len(raw)}, returning={len(result)}: {result}")
    return result


def reward_cabin_match(completions: list, **kwargs) -> list[float]:
    n = len(completions)
    raw = kwargs.get("cabin_match_reward", [])
    values = _broadcast(raw, n)
    result = [v * 0.15 for v in values]
    print(f"[REWARD cabin_match] completions={n}, raw={len(raw)}, returning={len(result)}: {result}")
    return result


def reward_group_integrity(completions: list, **kwargs) -> list[float]:
    n = len(completions)
    raw = kwargs.get("group_integrity_reward", [])
    values = _broadcast(raw, n)
    result = [v * 0.15 for v in values]
    print(f"[REWARD group_integrity] completions={n}, raw={len(raw)}, returning={len(result)}: {result}")
    return result


def reward_deadline(completions: list, **kwargs) -> list[float]:
    n = len(completions)
    raw = kwargs.get("deadline_reward", [])
    values = _broadcast(raw, n)
    result = [v * 0.15 for v in values]
    print(f"[REWARD deadline] completions={n}, raw={len(raw)}, returning={len(result)}: {result}")
    return result


REWARD_FUNCS = [
    reward_coverage,
    reward_ssr_integrity,
    reward_cabin_match,
    reward_group_integrity,
    reward_deadline,
]
