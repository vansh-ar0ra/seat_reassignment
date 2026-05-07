"""Training-side reward shaping and advantage computation.

Weights are applied inside each function so the GRPOTrainer sums them
directly (no ``reward_weights`` kwarg needed).
"""


def reward_coverage(completions: list, **kwargs) -> list[float]:
    raw = kwargs.get("coverage_reward", [0.0] * len(completions))
    result = [v * 0.35 for v in raw]
    print(f"[REWARD coverage] completions={len(completions)}, raw={len(raw)}, returning={len(result)}: {result}")
    return result


def reward_ssr_integrity(completions: list, **kwargs) -> list[float]:
    raw = kwargs.get("ssr_integrity_reward", [0.0] * len(completions))
    result = [v * 0.20 for v in raw]
    print(f"[REWARD ssr_integrity] completions={len(completions)}, raw={len(raw)}, returning={len(result)}: {result}")
    return result


def reward_cabin_match(completions: list, **kwargs) -> list[float]:
    raw = kwargs.get("cabin_match_reward", [0.0] * len(completions))
    result = [v * 0.15 for v in raw]
    print(f"[REWARD cabin_match] completions={len(completions)}, raw={len(raw)}, returning={len(result)}: {result}")
    return result


def reward_group_integrity(completions: list, **kwargs) -> list[float]:
    raw = kwargs.get("group_integrity_reward", [0.0] * len(completions))
    result = [v * 0.15 for v in raw]
    print(f"[REWARD group_integrity] completions={len(completions)}, raw={len(raw)}, returning={len(result)}: {result}")
    return result


def reward_deadline(completions: list, **kwargs) -> list[float]:
    raw = kwargs.get("deadline_reward", [0.0] * len(completions))
    result = [v * 0.15 for v in raw]
    print(f"[REWARD deadline] completions={len(completions)}, raw={len(raw)}, returning={len(result)}: {result}")
    return result


REWARD_FUNCS = [
    reward_coverage,
    reward_ssr_integrity,
    reward_cabin_match,
    reward_group_integrity,
    reward_deadline,
]
