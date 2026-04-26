"""CPU-only unit tests for training/dataset.py — the dataset builder.

Verifies:
  - Correct number of rows generated
  - Required columns present (prompt, seed, tier, task_id)
  - Tier distribution matches inputs
  - task_id format is correct (e.g. "easy_000", "medium_001")
  - Seeds are unique and sequential
  - base_seed offset works correctly
  - Edge case: all zeros → empty dataset
  - prompt column is a non-empty string

No GPU or torch needed (only depends on `datasets` library).
"""

import pytest
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    from training.dataset import build_dataset, _PROMPT
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

pytestmark = pytest.mark.skipif(not HAS_DATASETS, reason="datasets library not installed")


class TestBuildDataset:
    def test_default_total_rows(self):
        ds = build_dataset()
        assert len(ds) == 250  # 50 + 100 + 100

    def test_custom_counts(self):
        ds = build_dataset(n_easy=3, n_medium=5, n_hard=7)
        assert len(ds) == 15

    def test_required_columns(self):
        ds = build_dataset(n_easy=2, n_medium=2, n_hard=2)
        assert "prompt" in ds.column_names
        assert "seed" in ds.column_names
        assert "tier" in ds.column_names
        assert "task_id" in ds.column_names

    def test_tier_distribution(self):
        ds = build_dataset(n_easy=3, n_medium=5, n_hard=7)
        tiers = ds["tier"]
        assert tiers.count("easy") == 3
        assert tiers.count("medium") == 5
        assert tiers.count("hard") == 7

    def test_task_id_format(self):
        ds = build_dataset(n_easy=2, n_medium=2, n_hard=2)
        task_ids = ds["task_id"]
        assert task_ids[0] == "easy_000"
        assert task_ids[1] == "easy_001"
        assert task_ids[2] == "medium_000"
        assert task_ids[3] == "medium_001"
        assert task_ids[4] == "hard_000"
        assert task_ids[5] == "hard_001"

    def test_seeds_are_sequential(self):
        ds = build_dataset(n_easy=3, n_medium=2, n_hard=1, base_seed=0)
        seeds = ds["seed"]
        assert seeds == [0, 1, 2, 3, 4, 5]

    def test_base_seed_offset(self):
        ds = build_dataset(n_easy=2, n_medium=0, n_hard=0, base_seed=100)
        seeds = ds["seed"]
        assert seeds == [100, 101]

    def test_all_prompts_are_constant(self):
        ds = build_dataset(n_easy=5, n_medium=5, n_hard=5)
        for prompt in ds["prompt"]:
            assert prompt == _PROMPT

    def test_prompt_is_nonempty_string(self):
        assert isinstance(_PROMPT, str)
        assert len(_PROMPT) > 10

    def test_zero_counts(self):
        ds = build_dataset(n_easy=0, n_medium=0, n_hard=0)
        assert len(ds) == 0

    def test_single_tier(self):
        ds = build_dataset(n_easy=5, n_medium=0, n_hard=0)
        assert len(ds) == 5
        assert all(t == "easy" for t in ds["tier"])

    def test_seeds_unique(self):
        ds = build_dataset(n_easy=10, n_medium=10, n_hard=10)
        seeds = ds["seed"]
        assert len(set(seeds)) == len(seeds), "Seeds must be unique"
