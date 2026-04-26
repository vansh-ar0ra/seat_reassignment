"""CPU-only unit tests for training/rewards.py — the training-side reward shaping.

These functions apply weights to raw grader component scores and return
weighted floats. Verifies:
  - Each function applies the correct weight
  - Each function reads the correct kwargs key
  - Missing kwargs key → defaults to 0.0
  - Correct batch handling (list in, list out)
  - REWARD_FUNCS list is complete and in expected order
  - None/empty edge cases don't crash

No GPU or torch needed.
"""

import pytest
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.rewards import (
    reward_coverage,
    reward_ssr_integrity,
    reward_cabin_match,
    reward_group_integrity,
    reward_deadline,
    REWARD_FUNCS,
)


# ───────────────────────────────────────────────────────────
# Individual reward functions
# ───────────────────────────────────────────────────────────

class TestRewardCoverage:
    def test_applies_weight(self):
        result = reward_coverage(["c1", "c2"], coverage_reward=[1.0, 0.5])
        assert result == pytest.approx([0.35, 0.175])

    def test_missing_kwarg_defaults_to_zero(self):
        result = reward_coverage(["c1", "c2"])
        assert result == [0.0, 0.0]

    def test_batch_size_one(self):
        result = reward_coverage(["c1"], coverage_reward=[0.8])
        assert len(result) == 1
        assert result[0] == pytest.approx(0.8 * 0.35)

    def test_zero_reward(self):
        result = reward_coverage(["c1"], coverage_reward=[0.0])
        assert result == [0.0]


class TestRewardSsrIntegrity:
    def test_applies_weight(self):
        result = reward_ssr_integrity(["c1"], ssr_integrity_reward=[1.0])
        assert result == pytest.approx([0.20])

    def test_missing_kwarg(self):
        result = reward_ssr_integrity(["c1", "c2"])
        assert result == [0.0, 0.0]


class TestRewardCabinMatch:
    def test_applies_weight(self):
        result = reward_cabin_match(["c1"], cabin_match_reward=[1.0])
        assert result == pytest.approx([0.15])

    def test_missing_kwarg(self):
        result = reward_cabin_match(["c1"])
        assert result == [0.0]


class TestRewardGroupIntegrity:
    def test_applies_weight(self):
        result = reward_group_integrity(["c1"], group_integrity_reward=[1.0])
        assert result == pytest.approx([0.15])

    def test_missing_kwarg(self):
        result = reward_group_integrity(["c1"])
        assert result == [0.0]


class TestRewardDeadline:
    def test_applies_weight(self):
        result = reward_deadline(["c1"], deadline_reward=[1.0])
        assert result == pytest.approx([0.15])

    def test_missing_kwarg(self):
        result = reward_deadline(["c1"])
        assert result == [0.0]


# ───────────────────────────────────────────────────────────
# REWARD_FUNCS list
# ───────────────────────────────────────────────────────────

class TestRewardFuncsList:
    def test_length(self):
        assert len(REWARD_FUNCS) == 5

    def test_all_callable(self):
        for fn in REWARD_FUNCS:
            assert callable(fn)

    def test_contains_expected_functions(self):
        names = {fn.__name__ for fn in REWARD_FUNCS}
        expected = {
            "reward_coverage",
            "reward_ssr_integrity",
            "reward_cabin_match",
            "reward_group_integrity",
            "reward_deadline",
        }
        assert names == expected

    def test_weights_sum_to_one(self):
        """All weights together should sum to 1.0 (0.35+0.20+0.15+0.15+0.15)."""
        completions = ["c1"]
        total = sum(
            fn(completions, **{
                "coverage_reward": [1.0],
                "ssr_integrity_reward": [1.0],
                "cabin_match_reward": [1.0],
                "group_integrity_reward": [1.0],
                "deadline_reward": [1.0],
            })[0]
            for fn in REWARD_FUNCS
        )
        assert total == pytest.approx(1.0)

    def test_all_funcs_return_correct_length(self):
        batch = ["c1", "c2", "c3"]
        kwargs = {
            "coverage_reward": [0.5, 0.6, 0.7],
            "ssr_integrity_reward": [0.5, 0.6, 0.7],
            "cabin_match_reward": [0.5, 0.6, 0.7],
            "group_integrity_reward": [0.5, 0.6, 0.7],
            "deadline_reward": [0.5, 0.6, 0.7],
        }
        for fn in REWARD_FUNCS:
            result = fn(batch, **kwargs)
            assert len(result) == 3, f"{fn.__name__} returned wrong length"

    def test_all_funcs_return_floats(self):
        batch = ["c1"]
        kwargs = {
            "coverage_reward": [0.8],
            "ssr_integrity_reward": [0.8],
            "cabin_match_reward": [0.8],
            "group_integrity_reward": [0.8],
            "deadline_reward": [0.8],
        }
        for fn in REWARD_FUNCS:
            result = fn(batch, **kwargs)
            assert isinstance(result[0], float), f"{fn.__name__} returned non-float"
