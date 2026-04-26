"""CPU-only unit tests for message formatting and fallback logic in rollout.py.

Covers:
  - _format_main_task: returns a non-empty string
  - _format_state: includes step count, booked count, plan status
  - _format_instruction: correct text per step/plan state
  - _format_result: includes reward, tool result
  - _fallback_action: deterministic fallback per observation state

No GPU or torch needed.
"""

import pytest
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.rollout import (
    _format_main_task,
    _format_state,
    _format_instruction,
    _format_result,
    _fallback_action,
    SYSTEM_PROMPT,
    REWARD_COMPONENTS,
    _BREAKDOWN_KEY_MAP,
)


# ───────────────────────────────────────────────────────────
# Helpers — fake observation objects
# ───────────────────────────────────────────────────────────

def _make_obs(step_count=0, plan_submitted=False, booked=0, total=10,
              booked_summary=None, max_steps=5):
    return SimpleNamespace(
        step_count=step_count,
        max_steps=max_steps,
        passengers_booked=booked,
        passengers_total=total,
        passengers_remaining=total - booked,
        plan_submitted=plan_submitted,
        booked_summary=booked_summary or [],
    )


# ───────────────────────────────────────────────────────────
# _format_main_task
# ───────────────────────────────────────────────────────────

class TestFormatMainTask:
    def test_returns_nonempty(self):
        assert len(_format_main_task()) > 0

    def test_mentions_rebook(self):
        text = _format_main_task()
        assert "rebook" in text.lower() or "cancelled" in text.lower()


# ───────────────────────────────────────────────────────────
# _format_state
# ───────────────────────────────────────────────────────────

class TestFormatState:
    def test_includes_step_count(self):
        obs = _make_obs(step_count=2, max_steps=5)
        text = _format_state(obs)
        assert "2/5" in text

    def test_includes_booked_count(self):
        obs = _make_obs(booked=3, total=10)
        text = _format_state(obs)
        assert "3/10" in text or "Booked: 3" in text

    def test_includes_plan_status(self):
        obs = _make_obs(plan_submitted=True)
        text = _format_state(obs)
        assert "True" in text

    def test_shows_bookings_when_present(self):
        obs = _make_obs(
            booked=1,
            total=1,
            booked_summary=[{
                "passenger_id": "PAX-1",
                "flight_id": "FL-201",
                "cabin": "economy",
            }],
        )
        text = _format_state(obs)
        assert "PAX-1" in text
        assert "FL-201" in text


# ───────────────────────────────────────────────────────────
# _format_instruction
# ───────────────────────────────────────────────────────────

class TestFormatInstruction:
    def test_step_0(self):
        text = _format_instruction(0, plan_submitted=False)
        assert "get_full_manifest" in text

    def test_step_1(self):
        text = _format_instruction(1, plan_submitted=False)
        assert "get_flight_inventory" in text

    def test_step_2_no_plan(self):
        text = _format_instruction(2, plan_submitted=False)
        assert "reasoning" in text.lower() or "plan" in text.lower()

    def test_step_3_plan_submitted(self):
        text = _format_instruction(3, plan_submitted=True)
        assert "finalize" in text.lower()


# ───────────────────────────────────────────────────────────
# _format_result
# ───────────────────────────────────────────────────────────

class TestFormatResult:
    def test_includes_reward(self):
        text = _format_result(None, 0.5, "test reason")
        assert "0.5000" in text
        assert "test reason" in text

    def test_includes_tool_result(self):
        text = _format_result({"status": "success"}, 0.0, "ok")
        assert '"status": "success"' in text

    def test_none_tool_result(self):
        text = _format_result(None, 0.0, "ok")
        assert "Reward:" in text


# ───────────────────────────────────────────────────────────
# _fallback_action
# ───────────────────────────────────────────────────────────

class TestFallbackAction:
    def test_step_0_returns_manifest(self):
        obs = _make_obs(step_count=0, plan_submitted=False)
        action = _fallback_action(obs)
        assert action["tool_name"] == "get_full_manifest"

    def test_step_1_no_plan_returns_inventory(self):
        obs = _make_obs(step_count=1, plan_submitted=False)
        action = _fallback_action(obs)
        assert action["tool_name"] == "get_flight_inventory"

    def test_plan_submitted_returns_finalize(self):
        obs = _make_obs(step_count=3, plan_submitted=True)
        action = _fallback_action(obs)
        assert action["tool_name"] == "finalize_plan"

    def test_always_has_args(self):
        for step, plan_sub in [(0, False), (1, False), (3, True)]:
            obs = _make_obs(step_count=step, plan_submitted=plan_sub)
            action = _fallback_action(obs)
            assert "args" in action


# ───────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────

class TestConstants:
    def test_system_prompt_nonempty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_tools(self):
        assert "get_full_manifest" in SYSTEM_PROMPT
        assert "get_flight_inventory" in SYSTEM_PROMPT
        assert "submit_plan" in SYSTEM_PROMPT
        assert "finalize_plan" in SYSTEM_PROMPT

    def test_reward_components(self):
        assert len(REWARD_COMPONENTS) == 5
        assert "coverage_reward" in REWARD_COMPONENTS
        assert "ssr_integrity_reward" in REWARD_COMPONENTS

    def test_breakdown_key_map_covers_all_components(self):
        """Every reward component should be a value in _BREAKDOWN_KEY_MAP."""
        for component in REWARD_COMPONENTS:
            assert component in _BREAKDOWN_KEY_MAP.values()
