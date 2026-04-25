"""Unit tests for training/rollout.py.

Instantiates a mock trainer (with a real tokenizer), calls rollout_func on
2 prompts, and verifies the output dict has correct keys, shapes, and that
all 5 reward component keys are present.

The mock trainer uses a lightweight stub model to avoid needing a full
Qwen/Qwen3-4B-Instruct download during CI. For full integration tests with
a real model, run with --integration flag.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

# Reward component keys the rollout must produce
EXPECTED_REWARD_KEYS = {
    "coverage_reward",
    "cabin_match_reward",
    "group_integrity_reward",
    "deadline_reward",
    "ssr_integrity_reward",
}

# Required TRL dict keys
REQUIRED_KEYS = {"prompt_ids", "completion_ids", "logprobs"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal tokenizer that implements the interface rollout.py needs.

    Uses a real-ish encode/decode based on simple byte-level splitting so that
    token IDs are consistent and the chat template works.
    """

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self._next_id = 2

    def _get_id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_id
            self._next_id += 1
        return self._vocab[word]

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts = []
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Simple word-level tokenization
        words = text.split()
        return [self._get_id(w) for w in words]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        words = []
        for i in ids:
            w = inv.get(i, f"<unk_{i}>")
            if skip_special_tokens and w in ("<pad>", "<eos>"):
                continue
            words.append(w)
        return " ".join(words)

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs: Any) -> Any:
        import torch

        ids = self.encode(text)
        return {"input_ids": torch.tensor([ids])}


class _StubModel:
    """Stub model that produces a fixed, parseable action response."""

    _call_count: int = 0

    def __init__(self) -> None:
        self._call_count = 0
        import torch

        self._dummy_param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._dummy_param])

    def generate(self, input_ids, **kwargs):
        import torch

        self._call_count += 1

        # Produce a fixed response that the parser can handle.
        # Cycle through the 4-step episode: manifest -> inventory -> plan -> finalize
        step = (self._call_count - 1) % 4
        responses = [
            '<action>\n{"tool_name": "get_full_manifest", "args": {}}\n</action>',
            '<action>\n{"tool_name": "get_flight_inventory", "args": {}}\n</action>',
            (
                '<observations>\n8 passengers, 3 flights.\n</observations>\n'
                '<strategy>\nAssign all to FL-201.\n</strategy>\n'
                '<action>\n{"tool_name": "submit_plan", "args": {"PAX-E001": {"flight_id": "FL-201", "cabin": "business"}, "PAX-E002": {"flight_id": "FL-201", "cabin": "business"}, "PAX-E003": {"flight_id": "FL-201", "cabin": "business"}, "PAX-E004": {"flight_id": "FL-201", "cabin": "economy"}, "PAX-E005": {"flight_id": "FL-201", "cabin": "economy"}, "PAX-E006": {"flight_id": "FL-201", "cabin": "economy"}, "PAX-E007": {"flight_id": "FL-201", "cabin": "economy"}, "PAX-E008": {"flight_id": "FL-201", "cabin": "economy"}}}\n</action>'
            ),
            '<action>\n{"tool_name": "finalize_plan", "args": {}}\n</action>',
        ]
        response_text = responses[step]

        # Get tokenizer to encode the response
        tokenizer = _StubTokenizer()
        response_ids = tokenizer.encode(response_text)

        # Build full sequence: input_ids + response_ids
        full_ids = input_ids[0].tolist() + response_ids
        sequences = torch.tensor([full_ids])

        # Build scores (one per generated token)
        vocab_size = max(full_ids) + 100
        scores = []
        for tok_id in response_ids:
            logits = torch.full((1, vocab_size), -10.0)
            logits[0, tok_id] = 0.0  # Make the actual token the most likely
            scores.append(logits)

        return SimpleNamespace(sequences=sequences, scores=scores)

    def eval(self):
        return self


def _make_mock_trainer(num_generations: int = 1) -> Any:
    """Build a mock trainer with stub tokenizer and model."""
    trainer = MagicMock()
    trainer.processing_class = _StubTokenizer()
    trainer.model = _StubModel()
    trainer.args = SimpleNamespace(num_generations=num_generations)
    # No vLLM — use the fallback model.generate path
    trainer.vllm_generation = None
    return trainer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@needs_torch
class TestRolloutFunc:
    """Tests for the exported rollout_func."""

    def test_output_has_required_keys(self) -> None:
        """rollout_func returns all keys TRL needs."""
        from training.rollout import rollout_func

        trainer = _make_mock_trainer(num_generations=1)
        prompts = ["prompt_0", "prompt_1"]

        result = rollout_func(prompts, trainer)

        missing = REQUIRED_KEYS - result.keys()
        assert not missing, f"Missing required keys: {missing}"

    def test_output_has_env_mask(self) -> None:
        """rollout_func returns env_mask for completion masking."""
        from training.rollout import rollout_func

        trainer = _make_mock_trainer(num_generations=1)
        prompts = ["prompt_0", "prompt_1"]

        result = rollout_func(prompts, trainer)

        assert "env_mask" in result

    def test_batch_size_matches_prompts(self) -> None:
        """len(completion_ids) == len(prompts) * num_generations."""
        from training.rollout import rollout_func

        num_prompts = 2
        num_gen = 1
        trainer = _make_mock_trainer(num_generations=num_gen)
        prompts = [f"prompt_{i}" for i in range(num_prompts)]

        result = rollout_func(prompts, trainer)

        expected_count = num_prompts * num_gen
        assert len(result["completion_ids"]) == expected_count, (
            f"Expected {expected_count} completions, got {len(result['completion_ids'])}"
        )
        assert len(result["prompt_ids"]) == expected_count
        assert len(result["logprobs"]) == expected_count
        assert len(result["env_mask"]) == expected_count

    def test_completion_mask_matches_completion_shape(self) -> None:
        """env_mask[i] has the same length as completion_ids[i]."""
        from training.rollout import rollout_func

        trainer = _make_mock_trainer(num_generations=1)
        prompts = ["prompt_0", "prompt_1"]

        result = rollout_func(prompts, trainer)

        for i in range(len(result["completion_ids"])):
            assert len(result["env_mask"][i]) == len(result["completion_ids"][i]), (
                f"Sample {i}: env_mask len {len(result['env_mask'][i])} != "
                f"completion_ids len {len(result['completion_ids'][i])}"
            )

    def test_logprobs_matches_completion_shape(self) -> None:
        """logprobs[i] has the same length as completion_ids[i]."""
        from training.rollout import rollout_func

        trainer = _make_mock_trainer(num_generations=1)
        prompts = ["prompt_0"]

        result = rollout_func(prompts, trainer)

        for i in range(len(result["completion_ids"])):
            assert len(result["logprobs"][i]) == len(result["completion_ids"][i]), (
                f"Sample {i}: logprobs len {len(result['logprobs'][i])} != "
                f"completion_ids len {len(result['completion_ids'][i])}"
            )

    def test_all_reward_components_present(self) -> None:
        """All 5 reward component keys are returned."""
        from training.rollout import rollout_func

        trainer = _make_mock_trainer(num_generations=1)
        prompts = ["prompt_0", "prompt_1"]

        result = rollout_func(prompts, trainer)

        missing = EXPECTED_REWARD_KEYS - result.keys()
        assert not missing, f"Missing reward component keys: {missing}"

    def test_reward_components_correct_length(self) -> None:
        """Each reward component list has len == num_prompts * num_generations."""
        from training.rollout import rollout_func

        num_prompts = 2
        num_gen = 1
        trainer = _make_mock_trainer(num_generations=num_gen)
        prompts = [f"prompt_{i}" for i in range(num_prompts)]

        result = rollout_func(prompts, trainer)

        expected_count = num_prompts * num_gen
        for key in EXPECTED_REWARD_KEYS:
            assert len(result[key]) == expected_count, (
                f"{key}: expected {expected_count}, got {len(result[key])}"
            )

    def test_reward_components_are_floats(self) -> None:
        """Reward component values are floats."""
        from training.rollout import rollout_func

        trainer = _make_mock_trainer(num_generations=1)
        prompts = ["prompt_0"]

        result = rollout_func(prompts, trainer)

        for key in EXPECTED_REWARD_KEYS:
            for val in result[key]:
                assert isinstance(val, float), f"{key} contains non-float: {type(val)}"

    def test_env_mask_contains_only_0_and_1(self) -> None:
        """env_mask values are either 0 or 1."""
        from training.rollout import rollout_func

        trainer = _make_mock_trainer(num_generations=1)
        prompts = ["prompt_0"]

        result = rollout_func(prompts, trainer)

        for i, mask in enumerate(result["env_mask"]):
            unique = set(mask)
            assert unique <= {0, 1}, f"Sample {i}: env_mask has values {unique}"

    def test_env_mask_has_model_tokens(self) -> None:
        """env_mask has at least some model tokens (mask=1)."""
        from training.rollout import rollout_func

        trainer = _make_mock_trainer(num_generations=1)
        prompts = ["prompt_0"]

        result = rollout_func(prompts, trainer)

        for i, mask in enumerate(result["env_mask"]):
            assert any(v == 1 for v in mask), (
                f"Sample {i}: env_mask has no model tokens (all zeros)"
            )

    def test_prompt_ids_are_int_lists(self) -> None:
        """prompt_ids entries are lists of ints."""
        from training.rollout import rollout_func

        trainer = _make_mock_trainer(num_generations=1)
        prompts = ["prompt_0"]

        result = rollout_func(prompts, trainer)

        for i, pids in enumerate(result["prompt_ids"]):
            assert isinstance(pids, list), f"Sample {i}: prompt_ids is {type(pids)}"
            assert all(isinstance(x, int) for x in pids), (
                f"Sample {i}: prompt_ids contains non-int"
            )

    def test_num_generations_multiplies_output(self) -> None:
        """With num_generations=2, output length doubles."""
        from training.rollout import rollout_func

        num_prompts = 1
        num_gen = 2
        trainer = _make_mock_trainer(num_generations=num_gen)
        prompts = [f"prompt_{i}" for i in range(num_prompts)]

        result = rollout_func(prompts, trainer)

        expected = num_prompts * num_gen
        assert len(result["completion_ids"]) == expected


@needs_torch
class TestPlayEpisode:
    """Tests for the internal _play_episode helper."""

    def test_returns_all_keys(self) -> None:
        from training.rollout import _play_episode

        trainer = _make_mock_trainer()
        tokenizer = trainer.processing_class

        result = _play_episode(trainer, tokenizer, task_id="easy", seed=0)

        assert "prompt_ids" in result
        assert "completion_ids" in result
        assert "logprobs" in result
        assert "env_mask" in result
        assert "breakdown" in result
        assert "grader_score" in result

    def test_grader_score_non_negative(self) -> None:
        from training.rollout import _play_episode

        trainer = _make_mock_trainer()
        tokenizer = trainer.processing_class

        result = _play_episode(trainer, tokenizer, task_id="easy", seed=0)

        assert result["grader_score"] >= 0.0


class TestParseAction:
    """Tests for the action parser."""

    def test_clean_action_tags(self) -> None:
        from training.rollout import parse_action

        r = parse_action('<action>{"tool_name": "get_full_manifest", "args": {}}</action>')
        assert r is not None
        assert r["tool_name"] == "get_full_manifest"

    def test_nested_json(self) -> None:
        from training.rollout import parse_action

        text = '<action>{"tool_name": "submit_plan", "args": {"PAX-001": {"flight_id": "FL-201", "cabin": "economy"}}}</action>'
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "submit_plan"

    def test_malformed_returns_none(self) -> None:
        from training.rollout import parse_action

        r = parse_action("this is not valid")
        assert r is None

    def test_bare_json(self) -> None:
        from training.rollout import parse_action

        r = parse_action('{"tool_name": "finalize_plan", "args": {}}')
        assert r is not None
        assert r["tool_name"] == "finalize_plan"
