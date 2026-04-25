"""Training-side reward shaping and advantage computation."""


def reward_coverage(completions: list, **kwargs) -> list[float]:
    return kwargs.get("coverage_reward", [0.0] * len(completions))


def reward_ssr_integrity(completions: list, **kwargs) -> list[float]:
    return kwargs.get("ssr_integrity_reward", [0.0] * len(completions))


def reward_cabin_match(completions: list, **kwargs) -> list[float]:
    return kwargs.get("cabin_match_reward", [0.0] * len(completions))


def reward_group_integrity(completions: list, **kwargs) -> list[float]:
    return kwargs.get("group_integrity_reward", [0.0] * len(completions))


def reward_deadline(completions: list, **kwargs) -> list[float]:
    return kwargs.get("deadline_reward", [0.0] * len(completions))


REWARD_FUNCS = [
    reward_coverage,
    reward_ssr_integrity,
    reward_cabin_match,
    reward_group_integrity,
    reward_deadline,
]

REWARD_WEIGHTS = [0.35, 0.20, 0.15, 0.15, 0.15]
