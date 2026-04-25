"""
Reward and grader tests for the Flight Rebooking Environment (plan-then-commit model).

Tests cover:
  - Plan submission, duplicate submission, finalize rewards
  - Info call rewards
  - Invalid tool rewards
  - Grader sub-score logic (unchanged from original)
  - Priority weight lookup

All data is constructed inline — no file I/O.
Run with: pytest tests/test_rewards.py -v
"""

import pytest

from server.rewards import (
    EPS,
    REWARD_DUPLICATE_SUBMIT,
    REWARD_INVALID_TOOL,
    REWARD_NO_PLAN_FINALIZE,
    REWARD_PER_CALL_COST,
    RewardComputer,
    priority_weight,
)


@pytest.fixture
def rc():
    """Default RewardComputer for 15 passengers, 5 max steps."""
    return RewardComputer(total_passengers=15, max_steps=5)


# ---------------------------------------------------------------------------
# TestInfoRewards
# ---------------------------------------------------------------------------

class TestInfoRewards:
    def test_info_call_returns_per_call_cost(self, rc):
        reward, reason = rc.reward_for_info_call()
        assert reward == REWARD_PER_CALL_COST
        assert "Information" in reason


# ---------------------------------------------------------------------------
# TestPlanRewards
# ---------------------------------------------------------------------------

class TestPlanRewards:
    def test_plan_submission_reward_is_preview_plus_cost(self, rc):
        preview = 0.85
        reward, reason = rc.reward_for_plan_submission(preview)
        assert reward == pytest.approx(preview + REWARD_PER_CALL_COST)
        assert "0.8500" in reason

    def test_plan_submission_reward_zero_preview(self, rc):
        reward, _ = rc.reward_for_plan_submission(0.0)
        assert reward == pytest.approx(REWARD_PER_CALL_COST)

    def test_plan_submission_reward_perfect_preview(self, rc):
        reward, _ = rc.reward_for_plan_submission(1.0)
        assert reward == pytest.approx(1.0 + REWARD_PER_CALL_COST)

    def test_duplicate_submit_penalty(self, rc):
        reward, reason = rc.reward_for_duplicate_submit()
        assert reward == REWARD_DUPLICATE_SUBMIT
        assert "already submitted" in reason.lower()


# ---------------------------------------------------------------------------
# TestFinalizeRewards
# ---------------------------------------------------------------------------

class TestFinalizeRewards:
    def test_finalize_with_plan(self, rc):
        reward, reason = rc.reward_for_finalize(has_plan=True)
        assert reward == REWARD_PER_CALL_COST
        assert "finalized" in reason.lower()

    def test_finalize_without_plan(self, rc):
        reward, reason = rc.reward_for_finalize(has_plan=False)
        assert reward == REWARD_NO_PLAN_FINALIZE
        assert "without" in reason.lower()


# ---------------------------------------------------------------------------
# TestInvalidToolRewards
# ---------------------------------------------------------------------------

class TestInvalidToolRewards:
    def test_invalid_tool_penalty(self, rc):
        reward, reason = rc.reward_for_invalid_tool()
        assert reward == REWARD_INVALID_TOOL
        assert "Unrecognized" in reason


# ---------------------------------------------------------------------------
# TestGraderScore — grader_score() must return values in (EPS, 1-EPS)
# ---------------------------------------------------------------------------

class TestGraderScore:
    def test_perfect_score_easy(self):
        rc = RewardComputer(total_passengers=8, max_steps=5)
        passengers = {
            f"PAX-{i}": {
                "priority_tier": 3,
                "original_cabin": "economy",
                "group_id": None,
                "group_integrity": None,
                "ssr_flags": [],
                "downstream_deadline": None,
            }
            for i in range(8)
        }
        flights = {
            "FL-201": {
                "departure_time": "10:00",
                "arrival_time": "13:00",
                "supports_ssr": [],
                "cabin_availability": {"economy": 10},
            }
        }
        bookings = {
            pid: {"flight_id": "FL-201", "cabin": "economy"}
            for pid in passengers
        }

        score = rc.grader_score(bookings, passengers, flights, {})
        assert score > 0.99

    def test_zero_bookings_low_score(self):
        rc = RewardComputer(total_passengers=5, max_steps=5)
        passengers = {
            f"PAX-{i}": {
                "priority_tier": 3,
                "original_cabin": "economy",
                "group_id": None,
                "group_integrity": None,
                "ssr_flags": [],
                "downstream_deadline": None,
            }
            for i in range(5)
        }
        score = rc.grader_score({}, passengers, {}, {})
        assert score <= 0.5
        assert score >= EPS

    def test_grader_score_clamped_above_eps(self):
        rc = RewardComputer(total_passengers=1, max_steps=5)
        pax = {
            "PAX-1": {
                "priority_tier": 1,
                "original_cabin": "business",
                "group_id": None,
                "group_integrity": None,
                "ssr_flags": [],
                "downstream_deadline": None,
            }
        }
        score = rc.grader_score({}, pax, {}, {})
        assert score >= EPS

    def test_grader_score_clamped_below_one_minus_eps(self):
        rc = RewardComputer(total_passengers=1, max_steps=5)
        pax = {
            "PAX-1": {
                "priority_tier": 1,
                "original_cabin": "economy",
                "group_id": None,
                "group_integrity": None,
                "ssr_flags": [],
                "downstream_deadline": None,
            }
        }
        flights = {
            "FL-1": {
                "departure_time": "10:00",
                "arrival_time": "13:00",
                "supports_ssr": [],
                "cabin_availability": {"economy": 10},
            }
        }
        bookings = {"PAX-1": {"flight_id": "FL-1", "cabin": "economy"}}
        score = rc.grader_score(bookings, pax, flights, {})
        assert score <= 1.0 - EPS


# ---------------------------------------------------------------------------
# TestTerminalBreakdown — sub-score helpers
# ---------------------------------------------------------------------------

class TestTerminalBreakdown:
    def test_cabin_match_all_matched(self):
        rc = RewardComputer(total_passengers=2, max_steps=5)
        pax = {
            "A": {"priority_tier": 1, "original_cabin": "business",
                   "ssr_flags": [], "downstream_deadline": None,
                   "group_id": None, "group_integrity": None},
            "B": {"priority_tier": 3, "original_cabin": "economy",
                   "ssr_flags": [], "downstream_deadline": None,
                   "group_id": None, "group_integrity": None},
        }
        flights = {
            "FL": {"departure_time": "10:00", "arrival_time": "13:00",
                   "supports_ssr": [], "cabin_availability": {"business": 1, "economy": 1}},
        }
        bookings = {
            "A": {"flight_id": "FL", "cabin": "business"},
            "B": {"flight_id": "FL", "cabin": "economy"},
        }
        bd = rc.terminal_breakdown(bookings, pax, flights, {})
        assert bd["cabin_match_score"] == pytest.approx(1.0)

    def test_cabin_match_none(self):
        rc = RewardComputer(total_passengers=1, max_steps=5)
        pax = {
            "A": {"priority_tier": 1, "original_cabin": "business",
                   "ssr_flags": [], "downstream_deadline": None,
                   "group_id": None, "group_integrity": None},
        }
        flights = {
            "FL": {"departure_time": "10:00", "arrival_time": "13:00",
                   "supports_ssr": [], "cabin_availability": {"economy": 1}},
        }
        bookings = {"A": {"flight_id": "FL", "cabin": "economy"}}
        bd = rc.terminal_breakdown(bookings, pax, flights, {})
        assert bd["cabin_match_score"] == pytest.approx(0.0)

    def test_no_groups_gives_full_score(self):
        rc = RewardComputer(total_passengers=1, max_steps=5)
        pax = {"A": {"priority_tier": 1, "original_cabin": "economy",
                      "ssr_flags": [], "downstream_deadline": None,
                      "group_id": None, "group_integrity": None}}
        flights = {"FL": {"departure_time": "10:00", "arrival_time": "13:00",
                          "supports_ssr": [], "cabin_availability": {"economy": 1}}}
        bookings = {"A": {"flight_id": "FL", "cabin": "economy"}}
        bd = rc.terminal_breakdown(bookings, pax, flights, {})
        assert bd["group_integrity_score"] == pytest.approx(1.0)

    def test_ssr_integrity_all_good(self):
        rc = RewardComputer(total_passengers=1, max_steps=5)
        pax = {"A": {"priority_tier": 1, "original_cabin": "economy",
                      "ssr_flags": ["WCHR"], "downstream_deadline": None,
                      "group_id": None, "group_integrity": None}}
        flights = {"FL": {"departure_time": "10:00", "arrival_time": "13:00",
                          "supports_ssr": ["WCHR", "UM"],
                          "cabin_availability": {"economy": 1}}}
        bookings = {"A": {"flight_id": "FL", "cabin": "economy"}}
        bd = rc.terminal_breakdown(bookings, pax, flights, {})
        assert bd["ssr_integrity_score"] == pytest.approx(1.0)
        assert bd["hard_violations"] == 0

    def test_ssr_violation_penalised(self):
        rc = RewardComputer(total_passengers=1, max_steps=5)
        pax = {"A": {"priority_tier": 1, "original_cabin": "economy",
                      "ssr_flags": ["WCHR"], "downstream_deadline": None,
                      "group_id": None, "group_integrity": None}}
        flights = {"FL": {"departure_time": "10:00", "arrival_time": "13:00",
                          "supports_ssr": [],
                          "cabin_availability": {"economy": 1}}}
        bookings = {"A": {"flight_id": "FL", "cabin": "economy"}}
        bd = rc.terminal_breakdown(bookings, pax, flights, {})
        assert bd["ssr_integrity_score"] < 1.0
        assert bd["hard_violations"] >= 1


# ---------------------------------------------------------------------------
# TestPriorityWeights
# ---------------------------------------------------------------------------

class TestPriorityWeights:
    def test_known_weights(self):
        assert priority_weight(1) == 1.5
        assert priority_weight(2) == 1.3
        assert priority_weight(3) == 1.0
        assert priority_weight(4) == 0.8
        assert priority_weight(5) == 0.6

    def test_unknown_tier_default(self):
        assert priority_weight(99) == 1.0
