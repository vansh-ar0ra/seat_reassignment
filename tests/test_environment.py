"""
Integration tests for FlightRebookingEnvironment (plan-then-commit model).

Tests instantiate the environment directly — no server, no WebSocket.

Easy task:   8 passengers, no groups, no SSR, no deadlines
Medium task: 15 passengers, 2 groups (1 hard, 1 soft), 2 SSR, 2 deadlines
Hard task:   25 passengers, 4 groups (2 hard, 2 soft), 6 SSR, 5 deadlines

Run with: pytest tests/test_environment.py -v
"""

import pytest

from models import FlightRebookingAction, FlightRebookingObservation
from server.environment import FlightRebookingEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(task_id: str = "medium") -> FlightRebookingEnvironment:
    env = FlightRebookingEnvironment()
    env.reset(task_id=task_id)
    return env


def call_tool(env, tool_name, **args) -> FlightRebookingObservation:
    return env.step(FlightRebookingAction(tool_name=tool_name, args=args))


def submit_plan(env, assignments) -> FlightRebookingObservation:
    return call_tool(env, "submit_plan", assignments=assignments)


# ---------------------------------------------------------------------------
# Known-optimal assignments for each tier (grader ~ 1.0)
# ---------------------------------------------------------------------------

OPTIMAL_EASY = {
    "PAX-E001": {"flight_id": "FL-201", "cabin": "business"},
    "PAX-E002": {"flight_id": "FL-201", "cabin": "business"},
    "PAX-E003": {"flight_id": "FL-202", "cabin": "business"},
    "PAX-E004": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-E005": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-E006": {"flight_id": "FL-202", "cabin": "economy"},
    "PAX-E007": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-E008": {"flight_id": "FL-203", "cabin": "economy"},
}

OPTIMAL_MEDIUM = {
    "PAX-M001": {"flight_id": "FL-201", "cabin": "business"},
    "PAX-M002": {"flight_id": "FL-201", "cabin": "business"},
    "PAX-M003": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-M004": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-M005": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-M006": {"flight_id": "FL-201", "cabin": "premium_economy"},
    "PAX-M007": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-M008": {"flight_id": "FL-202", "cabin": "economy"},
    "PAX-M009": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-M010": {"flight_id": "FL-202", "cabin": "business"},
    "PAX-M011": {"flight_id": "FL-202", "cabin": "business"},
    "PAX-M012": {"flight_id": "FL-202", "cabin": "premium_economy"},
    "PAX-M013": {"flight_id": "FL-202", "cabin": "economy"},
    "PAX-M014": {"flight_id": "FL-203", "cabin": "economy"},
    "PAX-M015": {"flight_id": "FL-201", "cabin": "premium_economy"},
}

OPTIMAL_HARD = {
    "PAX-H001": {"flight_id": "FL-201", "cabin": "business"},
    "PAX-H002": {"flight_id": "FL-201", "cabin": "business"},
    "PAX-H003": {"flight_id": "FL-201", "cabin": "business"},
    "PAX-H004": {"flight_id": "FL-202", "cabin": "business"},
    "PAX-H005": {"flight_id": "FL-203", "cabin": "business"},
    "PAX-H006": {"flight_id": "FL-202", "cabin": "business"},
    "PAX-H007": {"flight_id": "FL-201", "cabin": "premium_economy"},
    "PAX-H008": {"flight_id": "FL-201", "cabin": "premium_economy"},
    "PAX-H009": {"flight_id": "FL-202", "cabin": "premium_economy"},
    "PAX-H010": {"flight_id": "FL-203", "cabin": "premium_economy"},
    "PAX-H011": {"flight_id": "FL-202", "cabin": "economy"},
    "PAX-H012": {"flight_id": "FL-202", "cabin": "economy"},
    "PAX-H013": {"flight_id": "FL-202", "cabin": "economy"},
    "PAX-H014": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-H015": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-H016": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-H017": {"flight_id": "FL-202", "cabin": "economy"},
    "PAX-H018": {"flight_id": "FL-203", "cabin": "economy"},
    "PAX-H019": {"flight_id": "FL-203", "cabin": "economy"},
    "PAX-H020": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-H021": {"flight_id": "FL-203", "cabin": "economy"},
    "PAX-H022": {"flight_id": "FL-204", "cabin": "economy"},
    "PAX-H023": {"flight_id": "FL-203", "cabin": "economy"},
    "PAX-H024": {"flight_id": "FL-201", "cabin": "economy"},
    "PAX-H025": {"flight_id": "FL-204", "cabin": "economy"},
}


# ---------------------------------------------------------------------------
# 1. TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_basic_fields(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")

        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.passengers_remaining == 15
        assert obs.passengers_total == 15
        assert obs.tool_result is None
        assert obs.reward_reason == "Episode started"
        assert obs.step_count == 0

    def test_no_bookings_initially(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.booked_summary == []
        assert obs.passengers_booked == 0

    def test_max_steps_set(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.max_steps == 5

    def test_unknown_task_raises(self):
        env = FlightRebookingEnvironment()
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent")

    def test_plan_not_submitted_initially(self):
        env = FlightRebookingEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.plan_submitted is False


# ---------------------------------------------------------------------------
# 2. TestGetFullManifest
# ---------------------------------------------------------------------------

class TestGetFullManifest:
    def test_returns_all_passengers(self):
        env = make_env("medium")
        obs = call_tool(env, "get_full_manifest")
        assert obs.tool_result["status"] == "success"
        assert len(obs.tool_result["passengers"]) == 15

    def test_returns_full_details(self):
        env = make_env("medium")
        obs = call_tool(env, "get_full_manifest")
        entry = obs.tool_result["passengers"][0]
        assert "passenger_id" in entry
        assert "priority_tier" in entry
        assert "original_cabin" in entry
        assert "group_id" in entry
        assert "ssr_flags" in entry
        assert "downstream_deadline" in entry
        assert "group_integrity" in entry
        assert "group_size" in entry

    def test_small_negative_reward(self):
        env = make_env("medium")
        obs = call_tool(env, "get_full_manifest")
        assert obs.reward < 0  # per_call_cost


# ---------------------------------------------------------------------------
# 3. TestGetFlightInventory
# ---------------------------------------------------------------------------

class TestGetFlightInventory:
    def test_returns_all_flights(self):
        env = make_env("medium")
        obs = call_tool(env, "get_flight_inventory")
        assert obs.tool_result["status"] == "success"
        assert len(obs.tool_result["flights"]) == 4

    def test_includes_ssr_support(self):
        env = make_env("medium")
        obs = call_tool(env, "get_flight_inventory")
        for fl in obs.tool_result["flights"]:
            assert "supports_ssr" in fl
            assert "cabin_availability" in fl

    def test_small_negative_reward(self):
        env = make_env("medium")
        obs = call_tool(env, "get_flight_inventory")
        assert obs.reward < 0  # per_call_cost


# ---------------------------------------------------------------------------
# 4. TestSubmitPlan
# ---------------------------------------------------------------------------

class TestSubmitPlan:
    def test_valid_plan_accepted(self):
        env = make_env("easy")
        obs = submit_plan(env, OPTIMAL_EASY)
        assert obs.tool_result["status"] == "success"
        assert obs.tool_result["accepted_count"] == 8
        assert obs.tool_result["rejected_count"] == 0
        assert obs.passengers_remaining == 0

    def test_invalid_passenger_rejected(self):
        env = make_env("easy")
        bad_plan = {
            "PAX-FAKE": {"flight_id": "FL-201", "cabin": "economy"},
            "PAX-E001": {"flight_id": "FL-201", "cabin": "business"},
        }
        obs = submit_plan(env, bad_plan)
        assert obs.tool_result["status"] == "success"
        assert obs.tool_result["rejected_count"] == 1
        assert obs.tool_result["accepted_count"] == 1

    def test_ssr_violation_rejected(self):
        """PAX-M001 has UM SSR. FL-202 does not support UM."""
        env = make_env("medium")
        plan = {"PAX-M001": {"flight_id": "FL-202", "cabin": "business"}}
        obs = submit_plan(env, plan)
        result = obs.tool_result
        rejected = [p for p in result["per_passenger"] if p["status"] == "rejected"]
        assert len(rejected) == 1
        assert "ssr" in rejected[0]["reason"].lower()

    def test_capacity_overflow_rejected(self):
        """FL-203 has only 2 business seats. Try to book 3."""
        env = make_env("easy")
        plan = {
            "PAX-E001": {"flight_id": "FL-203", "cabin": "business"},
            "PAX-E002": {"flight_id": "FL-203", "cabin": "business"},
            "PAX-E003": {"flight_id": "FL-203", "cabin": "business"},
        }
        obs = submit_plan(env, plan)
        assert obs.tool_result["accepted_count"] == 2
        assert obs.tool_result["rejected_count"] == 1

    def test_deadline_violation_rejected(self):
        """PAX-M006 has deadline 14:30. FL-204 arrives 18:15."""
        env = make_env("medium")
        plan = {"PAX-M006": {"flight_id": "FL-204", "cabin": "premium_economy"}}
        obs = submit_plan(env, plan)
        rejected = [p for p in obs.tool_result["per_passenger"] if p["status"] == "rejected"]
        assert len(rejected) == 1
        assert "deadline" in rejected[0]["reason"].lower()

    def test_hard_group_split_detected(self):
        """GRP-M01 (hard) split across flights should trigger constraint_violations."""
        env = make_env("medium")
        plan = {
            "PAX-M003": {"flight_id": "FL-201", "cabin": "economy"},
            "PAX-M004": {"flight_id": "FL-202", "cabin": "economy"},
            "PAX-M005": {"flight_id": "FL-201", "cabin": "economy"},
        }
        obs = submit_plan(env, plan)
        assert len(obs.tool_result["constraint_violations"]) > 0

    def test_preview_score_returned(self):
        env = make_env("easy")
        obs = submit_plan(env, OPTIMAL_EASY)
        assert "plan_score_preview" in obs.tool_result
        assert obs.tool_result["plan_score_preview"] > 0.9

    def test_duplicate_submit_rejected(self):
        env = make_env("easy")
        submit_plan(env, OPTIMAL_EASY)
        obs = submit_plan(env, OPTIMAL_EASY)
        assert obs.tool_result["status"] == "error"
        assert "already submitted" in obs.tool_result["message"].lower()


# ---------------------------------------------------------------------------
# 5. TestFinalizePlan
# ---------------------------------------------------------------------------

class TestFinalizePlan:
    def test_triggers_done(self):
        env = make_env("easy")
        obs = call_tool(env, "finalize_plan")
        assert obs.done is True

    def test_grader_score_present(self):
        env = make_env("easy")
        obs = call_tool(env, "finalize_plan")
        assert "grader_score" in obs.tool_result

    def test_grader_score_in_range(self):
        env = make_env("easy")
        obs = call_tool(env, "finalize_plan")
        score = obs.tool_result["grader_score"]
        assert 0.0 < score < 1.0

    def test_no_plan_finalize_low_score(self):
        """Finalizing without submitting a plan gives a low score."""
        env = make_env("easy")
        obs = call_tool(env, "finalize_plan")
        assert obs.reward < 0  # penalty for no plan

    def test_step_after_done_raises(self):
        env = make_env("easy")
        call_tool(env, "finalize_plan")
        with pytest.raises(RuntimeError, match="terminated"):
            call_tool(env, "get_full_manifest")


# ---------------------------------------------------------------------------
# 6. TestOptimalPlans
# ---------------------------------------------------------------------------

class TestOptimalPlans:
    def test_optimal_easy(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        submit_plan(env, OPTIMAL_EASY)
        obs = call_tool(env, "finalize_plan")
        assert obs.done is True
        score = obs.tool_result["grader_score"]
        assert score > 0.99

    def test_optimal_medium(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="medium")
        submit_plan(env, OPTIMAL_MEDIUM)
        obs = call_tool(env, "finalize_plan")
        assert obs.done is True
        score = obs.tool_result["grader_score"]
        # Group integrity caps at 0.7 per group, so ~0.955 is the ceiling
        assert score > 0.95

    def test_optimal_hard(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="hard")
        submit_plan(env, OPTIMAL_HARD)
        obs = call_tool(env, "finalize_plan")
        assert obs.done is True
        score = obs.tool_result["grader_score"]
        # Group integrity caps at 0.7 per group, so ~0.955 is the ceiling
        assert score > 0.95


# ---------------------------------------------------------------------------
# 7. TestInvalidTool
# ---------------------------------------------------------------------------

class TestInvalidTool:
    def test_unknown_tool_error(self):
        env = make_env("easy")
        obs = env.step(FlightRebookingAction(tool_name="teleport_passenger", args={}))
        assert obs.tool_result["status"] == "error"
        assert "teleport_passenger" in obs.tool_result["message"]

    def test_unknown_tool_gives_negative_reward(self):
        env = make_env("easy")
        obs = env.step(FlightRebookingAction(tool_name="nonexistent", args={}))
        assert obs.reward < 0


# ---------------------------------------------------------------------------
# 8. TestEpisodeCompletion
# ---------------------------------------------------------------------------

class TestEpisodeCompletion:
    def test_state_is_complete_after_finalize(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        submit_plan(env, OPTIMAL_EASY)
        call_tool(env, "finalize_plan")
        s = env.state
        assert s.is_complete is True
        assert s.passengers_booked == 8
        assert s.passengers_remaining == 0
        assert s.plan_submitted is True

    def test_timeout_terminates(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        obs = None
        for _ in range(5):
            obs = call_tool(env, "get_full_manifest")
        assert obs.done is True
        assert "timed out" in obs.reward_reason.lower()

    def test_grader_present_on_timeout(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        obs = None
        for _ in range(5):
            obs = call_tool(env, "get_full_manifest")
        assert "grader_score" in obs.tool_result

    def test_terminal_breakdown_keys(self):
        env = FlightRebookingEnvironment()
        env.reset(task_id="easy")
        submit_plan(env, OPTIMAL_EASY)
        obs = call_tool(env, "finalize_plan")
        bd = obs.tool_result["terminal_breakdown"]
        assert set(bd.keys()) == {
            "coverage_score", "cabin_match_score", "group_integrity_score",
            "deadline_score", "ssr_integrity_score", "hard_violations",
        }

    def test_cumulative_reward_accumulates(self):
        env = make_env("easy")
        obs1 = call_tool(env, "get_full_manifest")
        obs2 = call_tool(env, "get_flight_inventory")
        assert obs2.cumulative_reward == pytest.approx(obs1.reward + obs2.reward)
