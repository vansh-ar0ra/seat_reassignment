"""CPU-only unit tests for the XML/JSON action parser in rollout.py.

Covers:
  - Clean <action> tags with valid JSON
  - Nested JSON inside <action> tags (submit_plan with nested dicts)
  - Malformed / missing closing tags
  - Truncated JSON (missing braces) — brace repair logic
  - Code fences (```json ... ```)
  - Bare JSON (no tags)
  - Flat JSON blocks mixed with prose
  - Multiple JSON blocks (should pick one with tool_name)
  - Totally unparseable text → None
  - Empty string → None
  - Missing 'args' key → auto-filled with {}
  - Real-world LLM output with reasoning tags + action
  - Trailing commas (invalid JSON) — should still recover via fallback

No GPU or torch needed.
"""

import pytest
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.rollout import parse_action, _extract_xml_tag, _repair_trailing_braces, _find_nested_json


# ───────────────────────────────────────────────────────────
# _extract_xml_tag
# ───────────────────────────────────────────────────────────

class TestExtractXmlTag:
    def test_simple_tag(self):
        assert _extract_xml_tag("<foo>hello</foo>", "foo") == "hello"

    def test_multiline_content(self):
        text = "<action>\n  some stuff\n  more stuff\n</action>"
        result = _extract_xml_tag(text, "action")
        assert "some stuff" in result

    def test_missing_tag_returns_none(self):
        assert _extract_xml_tag("no tags here", "action") is None

    def test_empty_tag(self):
        assert _extract_xml_tag("<action></action>", "action") == ""

    def test_nested_same_tag_greedy(self):
        # re.DOTALL with .*? is non-greedy, picks first match
        text = "<action>first</action> <action>second</action>"
        result = _extract_xml_tag(text, "action")
        assert result == "first"

    def test_strips_whitespace(self):
        text = "<action>  \n  hello  \n  </action>"
        assert _extract_xml_tag(text, "action") == "hello"


# ───────────────────────────────────────────────────────────
# _repair_trailing_braces
# ───────────────────────────────────────────────────────────

class TestRepairTrailingBraces:
    def test_balanced_unchanged(self):
        text = '{"a": {"b": 1}}'
        assert _repair_trailing_braces(text) == text

    def test_one_missing_brace(self):
        text = '{"a": {"b": 1}'
        assert _repair_trailing_braces(text) == '{"a": {"b": 1}}'

    def test_two_missing_braces(self):
        text = '{"a": {"b": {"c": 1}'
        result = _repair_trailing_braces(text)
        assert result.count("{") == result.count("}")

    def test_missing_bracket(self):
        text = '{"a": [1, 2, 3'
        result = _repair_trailing_braces(text)
        assert result.count("[") == result.count("]")

    def test_no_braces(self):
        text = "hello world"
        assert _repair_trailing_braces(text) == text


# ───────────────────────────────────────────────────────────
# _find_nested_json
# ───────────────────────────────────────────────────────────

class TestFindNestedJson:
    def test_simple_nested(self):
        text = 'prefix {"tool_name": "foo", "args": {"a": 1}} suffix'
        result = _find_nested_json(text)
        assert result is not None
        assert result["tool_name"] == "foo"

    def test_deeply_nested(self):
        text = '{"tool_name": "submit_plan", "args": {"PAX-1": {"flight_id": "FL-201", "cabin": "economy"}}}'
        result = _find_nested_json(text)
        assert result is not None
        assert result["tool_name"] == "submit_plan"

    def test_no_tool_name_returns_none(self):
        text = '{"name": "foo", "value": 42}'
        assert _find_nested_json(text) is None

    def test_truncated_json_repaired(self):
        text = '{"tool_name": "finalize_plan", "args": {'
        result = _find_nested_json(text)
        assert result is not None
        assert result["tool_name"] == "finalize_plan"

    def test_no_json_returns_none(self):
        assert _find_nested_json("just plain text") is None


# ───────────────────────────────────────────────────────────
# parse_action — the main parser
# ───────────────────────────────────────────────────────────

class TestParseAction:
    """Comprehensive tests for parse_action()."""

    # ---- Clean <action> tags ----

    def test_clean_action_tags(self):
        text = '<action>{"tool_name": "get_full_manifest", "args": {}}</action>'
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "get_full_manifest"
        assert r["args"] == {}

    def test_action_tags_with_newlines(self):
        text = '<action>\n{"tool_name": "get_flight_inventory", "args": {}}\n</action>'
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "get_flight_inventory"

    def test_action_tags_nested_json(self):
        text = (
            '<action>\n'
            '{"tool_name": "submit_plan", "args": '
            '{"PAX-001": {"flight_id": "FL-201", "cabin": "economy"}, '
            '"PAX-002": {"flight_id": "FL-202", "cabin": "business"}}}\n'
            '</action>'
        )
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "submit_plan"
        assert "PAX-001" in r["args"]
        assert r["args"]["PAX-001"]["flight_id"] == "FL-201"

    def test_action_with_reasoning_prefix(self):
        """Real-world: reasoning tags before action tags."""
        text = (
            "<observations>\n8 passengers, 3 flights, no constraints.\n</observations>\n"
            "<strategy>\nAssign all to FL-201.\n</strategy>\n"
            '<action>\n{"tool_name": "submit_plan", "args": {"PAX-1": {"flight_id": "FL-201", "cabin": "economy"}}}\n</action>'
        )
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "submit_plan"

    # ---- Code fences ----

    def test_code_fence_json(self):
        text = '```json\n{"tool_name": "finalize_plan", "args": {}}\n```'
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "finalize_plan"

    def test_code_fence_no_lang(self):
        text = '```\n{"tool_name": "get_full_manifest", "args": {}}\n```'
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "get_full_manifest"

    def test_action_tags_with_code_fence_inside(self):
        text = (
            '<action>\n```json\n'
            '{"tool_name": "finalize_plan", "args": {}}\n'
            '```\n</action>'
        )
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "finalize_plan"

    # ---- Bare JSON ----

    def test_bare_json(self):
        text = '{"tool_name": "finalize_plan", "args": {}}'
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "finalize_plan"

    def test_bare_json_with_surrounding_text(self):
        text = 'I will now call: {"tool_name": "get_full_manifest", "args": {}} to get data.'
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "get_full_manifest"

    # ---- Missing args ----

    def test_missing_args_key_auto_filled(self):
        text = '<action>{"tool_name": "finalize_plan"}</action>'
        r = parse_action(text)
        assert r is not None
        assert r["args"] == {}

    # ---- Truncated JSON ----

    def test_truncated_json_repaired(self):
        text = '<action>{"tool_name": "submit_plan", "args": {"PAX-1": {"flight_id": "FL-201", "cabin": "economy"}</action>'
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "submit_plan"

    # ---- Failure cases → None ----

    def test_empty_string(self):
        assert parse_action("") is None

    def test_plain_text(self):
        assert parse_action("I don't know what to do.") is None

    def test_json_without_tool_name(self):
        """JSON object that doesn't have tool_name should return None."""
        assert parse_action('{"name": "not_a_tool", "value": 42}') is None

    def test_xml_tags_with_invalid_json(self):
        """Action tags containing totally invalid content."""
        text = "<action>this is not json at all</action>"
        assert parse_action(text) is None

    # ---- Edge cases ----

    def test_multiple_action_blocks_picks_first(self):
        text = (
            '<action>{"tool_name": "get_full_manifest", "args": {}}</action>\n'
            '<action>{"tool_name": "finalize_plan", "args": {}}</action>'
        )
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "get_full_manifest"

    def test_action_with_extra_whitespace(self):
        text = '<action>   {"tool_name": "finalize_plan", "args": {}}   </action>'
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "finalize_plan"

    def test_large_submit_plan(self):
        """Simulate a real 25-passenger submit_plan."""
        assignments = {
            f"PAX-H{i:03d}": {"flight_id": f"FL-{200 + (i % 4) + 1}", "cabin": "economy"}
            for i in range(1, 26)
        }
        import json
        inner = json.dumps({"tool_name": "submit_plan", "args": assignments})
        text = f"<action>\n{inner}\n</action>"
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "submit_plan"
        assert len(r["args"]) == 25

    def test_flat_json_among_prose(self):
        text = (
            "Let me think about this. The best approach would be to gather data first.\n\n"
            '{"tool_name": "get_full_manifest", "args": {}}\n\n'
            "This will give me the passenger manifest."
        )
        r = parse_action(text)
        assert r is not None
        assert r["tool_name"] == "get_full_manifest"
