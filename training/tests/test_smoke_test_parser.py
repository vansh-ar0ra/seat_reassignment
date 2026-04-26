"""CPU-only unit tests for the parser in training/smoke_test.py.

The smoke_test.py has its own copy of the parser (parse_llm_response) and
formatting functions. We test these independently to ensure they stay in sync
with rollout.py's parse_action and catch regressions.

No GPU or torch needed.
"""

import pytest
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.smoke_test import (
    parse_llm_response,
    extract_xml_tag,
    format_main_task,
    format_state,
    format_instruction,
    format_result,
    fallback_action,
)
from types import SimpleNamespace


def _make_obs(**kwargs):
    defaults = dict(
        step_count=0, max_steps=5, passengers_booked=0, passengers_total=10,
        passengers_remaining=10, plan_submitted=False, booked_summary=[],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ───────────────────────────────────────────────────────────
# parse_llm_response — mirror tests from test_parser.py
# ───────────────────────────────────────────────────────────

class TestParseLlmResponse:
    def test_clean_action_tags(self):
        r = parse_llm_response('<action>{"tool_name": "get_full_manifest", "args": {}}</action>')
        assert r is not None
        assert r["tool_name"] == "get_full_manifest"

    def test_nested_json(self):
        text = '<action>{"tool_name": "submit_plan", "args": {"PAX-1": {"flight_id": "FL-201", "cabin": "economy"}}}</action>'
        r = parse_llm_response(text)
        assert r is not None
        assert r["tool_name"] == "submit_plan"

    def test_returns_none_on_garbage(self):
        assert parse_llm_response("completely unparseable gibberish") is None

    def test_bare_json(self):
        r = parse_llm_response('{"tool_name": "finalize_plan", "args": {}}')
        assert r is not None
        assert r["tool_name"] == "finalize_plan"

    def test_code_fence(self):
        r = parse_llm_response('```json\n{"tool_name": "finalize_plan", "args": {}}\n```')
        assert r is not None

    def test_empty_string(self):
        assert parse_llm_response("") is None

    def test_missing_args_key(self):
        r = parse_llm_response('<action>{"tool_name": "finalize_plan"}</action>')
        assert r is not None
        assert r["args"] == {}


# ───────────────────────────────────────────────────────────
# Parsers in sync: both parsers should agree on key inputs
# ───────────────────────────────────────────────────────────

class TestParserSync:
    """Verify that rollout.parse_action and smoke_test.parse_llm_response agree."""

    INPUTS = [
        '<action>{"tool_name": "get_full_manifest", "args": {}}</action>',
        '<action>{"tool_name": "submit_plan", "args": {"PAX-1": {"flight_id": "FL-201", "cabin": "eco"}}}</action>',
        '{"tool_name": "finalize_plan", "args": {}}',
        "not parseable at all",
        "",
        '<action>{"tool_name": "finalize_plan"}</action>',
    ]

    def test_both_agree(self):
        from training.rollout import parse_action

        for text in self.INPUTS:
            r1 = parse_action(text)
            r2 = parse_llm_response(text)
            if r1 is None:
                assert r2 is None, f"Disagreement on: {text!r}"
            else:
                assert r2 is not None, f"Disagreement on: {text!r}"
                assert r1["tool_name"] == r2["tool_name"], f"tool_name mismatch on: {text!r}"


# ───────────────────────────────────────────────────────────
# Formatting functions
# ───────────────────────────────────────────────────────────

class TestSmokeTestFormatting:
    def test_format_main_task(self):
        assert len(format_main_task()) > 0

    def test_format_state(self):
        obs = _make_obs(step_count=2)
        text = format_state(obs)
        assert "2/5" in text

    def test_format_instruction_step0(self):
        text = format_instruction(0, False)
        assert "get_full_manifest" in text

    def test_format_result_with_tool(self):
        text = format_result({"status": "ok"}, 0.5, "test")
        assert "0.5000" in text

    def test_fallback_step0(self):
        obs = _make_obs(step_count=0)
        assert fallback_action(obs)["tool_name"] == "get_full_manifest"

    def test_fallback_plan_submitted(self):
        obs = _make_obs(step_count=3, plan_submitted=True)
        assert fallback_action(obs)["tool_name"] == "finalize_plan"
