"""
Integration tests for SeatReassignmentEnvironment.

Tests instantiate the environment directly — no server, no WebSocket.
All passengers and their AC-1 seat assignments are fixed by the static data files.

Medium task (20 passengers, AC-1 seats rows 1-4):
  Business (rows 1-2):  PAX-001→1A, PAX-002→1B, PAX-003→1C, PAX-004→1D
                        PAX-005→2A, PAX-006→2B, PAX-007→2C, PAX-008→2D
  Economy  (rows 3-4):  PAX-009→3A, PAX-010→3B, PAX-011→3C, PAX-012→3D
                        PAX-013→3E, PAX-014→3F, PAX-015→4A, PAX-016→4B
                        PAX-017→4C, PAX-018→4D, PAX-019→4E, PAX-020→4F

Easy task (8 passengers, AC-1 seats rows 1-2):
  Business (row 1): PAX-E001→1A, PAX-E002→1B, PAX-E003→1C, PAX-E004→1D
  Economy  (row 2): PAX-E005→2A, PAX-E006→2B, PAX-E007→2C, PAX-E008→2D

Run with: pytest tests/test_environment.py -v
"""

import pytest

from models import SeatReassignmentAction, SeatReassignmentObservation
from server.environment import SeatReassignmentEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(task_id: str = "medium") -> SeatReassignmentEnvironment:
    env = SeatReassignmentEnvironment()
    env.reset(task_id=task_id)
    return env


def fetch(env, seat_id: str) -> SeatReassignmentObservation:
    return env.step(SeatReassignmentAction(tool_name="get_passenger_details", args={"seat_id": seat_id}))


def assign(env, passenger_id: str, target_seat_id: str) -> SeatReassignmentObservation:
    return env.step(SeatReassignmentAction(
        tool_name="assign_seat",
        args={"passenger_id": passenger_id, "target_seat_id": target_seat_id},
    ))


def swap(env, pid1: str, pid2: str) -> SeatReassignmentObservation:
    return env.step(SeatReassignmentAction(
        tool_name="swap_seats",
        args={"passenger_id_1": pid1, "passenger_id_2": pid2},
    ))


# ---------------------------------------------------------------------------
# 1. Reset produces a valid initial observation (medium task)
# ---------------------------------------------------------------------------

class TestReset:
    def test_basic_fields(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="medium")

        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.passengers_remaining == 20
        assert obs.passengers_total == 20
        assert obs.tool_result is None
        assert obs.reward_reason == "Episode started"
        assert obs.step_count == 0

    def test_all_ac1_seats_occupied(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="medium")
        assert len(obs.ac1_seats_occupied) == 20

    def test_ac2_starts_empty(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.ac2_seat_assignments == {}
        assert len(obs.ac2_seats_available) == 24  # AC-2 medium has 24 seats

    def test_layouts_present(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.ac1_layout.get("aircraft_id") == "AC-1"
        assert obs.ac2_layout.get("aircraft_id") == "AC-2"

    def test_max_steps_is_three_times_passengers(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="medium")
        assert obs.max_steps == 60  # 3 × 20

    def test_unknown_task_raises(self):
        env = SeatReassignmentEnvironment()
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(task_id="nonexistent")


# ---------------------------------------------------------------------------
# 2. Fetch then assign — happy path (medium task)
# ---------------------------------------------------------------------------

class TestFetchThenAssign:
    def test_fetch_returns_passenger_info(self):
        env = make_env()
        obs = fetch(env, "1A")

        assert obs.tool_result["status"] == "success"
        assert obs.tool_result["passenger_id"] == "PAX-001"
        assert obs.tool_result["cabin"] == "business"
        assert obs.tool_result["current_seat_type"] == "window"
        assert obs.tool_result["paid_window"] is True

    def test_assign_decrements_remaining(self):
        env = make_env()
        fetch(env, "1A")
        obs = assign(env, "PAX-001", "1A")  # AC-2 seat 1A (business window)

        assert obs.tool_result["status"] == "success"
        assert obs.passengers_remaining == 19
        assert "1A" in obs.ac2_seat_assignments
        assert obs.ac2_seat_assignments["1A"] == "PAX-001"

    def test_assigned_seat_removed_from_ac2_available(self):
        env = make_env()
        assign(env, "PAX-001", "1A")
        obs = assign(env, "PAX-002", "1B")

        assert "1A" not in obs.ac2_seats_available
        assert "1B" not in obs.ac2_seats_available

    def test_assigned_seat_removed_from_ac1_occupied(self):
        env = make_env()
        obs = assign(env, "PAX-001", "1A")
        assert "1A" not in obs.ac1_seats_occupied

    def test_reward_is_positive_for_valid_assignment(self):
        env = make_env()
        obs = assign(env, "PAX-001", "1A")  # cabin match + paid_window in window
        assert obs.reward > 0

    def test_state_reflects_assignment(self):
        env = make_env()
        assign(env, "PAX-001", "1A")
        s = env.state
        assert s.passengers_assigned == 1
        assert s.passengers_remaining == 19

    def test_cumulative_reward_accumulates(self):
        env = make_env()
        obs1 = assign(env, "PAX-001", "1A")
        obs2 = assign(env, "PAX-002", "1B")
        assert obs2.cumulative_reward == pytest.approx(obs1.reward + obs2.reward)




# ---------------------------------------------------------------------------
# 3. Invalid tool name
# ---------------------------------------------------------------------------

class TestInvalidTool:
    def test_unknown_tool_returns_error(self):
        env = make_env()
        obs = env.step(SeatReassignmentAction(tool_name="teleport_passenger", args={}))

        assert obs.tool_result["status"] == "error"
        assert "teleport_passenger" in obs.tool_result["message"]

    def test_unknown_tool_gives_negative_reward(self):
        env = make_env()
        obs = env.step(SeatReassignmentAction(tool_name="nonexistent", args={}))
        assert obs.reward < 0


# ---------------------------------------------------------------------------
# 4. Double booking
# ---------------------------------------------------------------------------

class TestDoubleBooking:
    def test_second_assignment_to_occupied_seat_errors(self):
        env = make_env()
        assign(env, "PAX-001", "3A")  # economy passenger to economy seat
        obs = assign(env, "PAX-009", "3A")  # try to take same seat

        assert obs.tool_result["status"] == "error"
        assert "occupied" in obs.tool_result["message"].lower()

    def test_passengers_remaining_unchanged_after_failed_assign(self):
        env = make_env()
        assign(env, "PAX-001", "1A")
        obs = assign(env, "PAX-002", "1A")  # occupied

        assert obs.tool_result["status"] == "error"
        assert obs.passengers_remaining == 19  # only PAX-001 was assigned


# ---------------------------------------------------------------------------
# 5. Swap
# ---------------------------------------------------------------------------

class TestSwap:
    def _setup_two_assigned(self):
        """Assign PAX-001 to 1B (aisle) and PAX-004 to 1D (window), then swap."""
        env = make_env()
        assign(env, "PAX-001", "1B")   # PAX-001 is paid_window, gets aisle
        assign(env, "PAX-004", "1D")   # PAX-004 is paid_window, gets window
        return env

    def test_swap_succeeds(self):
        env = self._setup_two_assigned()
        obs = swap(env, "PAX-001", "PAX-004")
        assert obs.tool_result["status"] == "success"

    def test_swap_changes_seats_in_observation(self):
        env = self._setup_two_assigned()
        swap(env, "PAX-001", "PAX-004")
        obs = env.step(SeatReassignmentAction(tool_name="get_passenger_details", args={"seat_id": "1A"}))
        # After swap: PAX-001 should be in 1D, PAX-004 in 1B
        assert obs.ac2_seat_assignments.get("1D") == "PAX-001"
        assert obs.ac2_seat_assignments.get("1B") == "PAX-004"

    def test_swap_not_yet_assigned_errors(self):
        env = make_env()
        assign(env, "PAX-001", "1A")
        obs = swap(env, "PAX-001", "PAX-002")  # PAX-002 not on AC-2 yet
        assert obs.tool_result["status"] == "error"
        assert "not yet assigned" in obs.tool_result["message"].lower()

    def test_swap_same_passenger_errors(self):
        env = make_env()
        assign(env, "PAX-001", "1A")
        obs = swap(env, "PAX-001", "PAX-001")
        assert obs.tool_result["status"] == "error"
        assert "themselves" in obs.tool_result["message"].lower()


# ---------------------------------------------------------------------------
# 6. Episode completion — all 20 passengers assigned (medium task)
# ---------------------------------------------------------------------------

# Known valid assignments that respect cabin constraints
FULL_ASSIGNMENTS_MEDIUM = {
    # Business passengers → AC-2 business seats
    "PAX-001": "1A", "PAX-002": "1B", "PAX-003": "1C", "PAX-004": "1D",
    "PAX-005": "2A", "PAX-006": "2B", "PAX-007": "2C", "PAX-008": "2D",
    # Economy passengers → AC-2 economy seats
    "PAX-009": "3A", "PAX-010": "3B", "PAX-011": "3C", "PAX-012": "3D",
    "PAX-013": "3E", "PAX-014": "3F", "PAX-015": "4A", "PAX-016": "4B",
    "PAX-017": "4C", "PAX-018": "4D", "PAX-019": "4E", "PAX-020": "4F",
}


class TestEpisodeCompletion:
    def _run_full_episode(self) -> SeatReassignmentObservation:
        env = SeatReassignmentEnvironment()
        env.reset(task_id="medium")
        obs = None
        for pid, seat in FULL_ASSIGNMENTS_MEDIUM.items():
            obs = assign(env, pid, seat)
        return obs

    def test_done_after_all_assigned(self):
        obs = self._run_full_episode()
        assert obs.done is True

    def test_passengers_remaining_is_zero(self):
        obs = self._run_full_episode()
        assert obs.passengers_remaining == 0

    def test_terminal_reward_included(self):
        obs = self._run_full_episode()
        assert obs.reward is not None
        assert obs.reward > 0

    def test_grader_score_in_observation(self):
        obs = self._run_full_episode()
        assert obs.tool_result is not None
        assert "grader_score" in obs.tool_result
        assert 0.0 <= obs.tool_result["grader_score"] <= 1.0

    def test_terminal_breakdown_present(self):
        obs = self._run_full_episode()
        assert "terminal_breakdown" in obs.tool_result
        bd = obs.tool_result["terminal_breakdown"]
        assert set(bd.keys()) == {
            "cabin_score", "preference_score", "efficiency_score",
            "weighted_total", "incomplete_penalty",
        }

    def test_perfect_cabin_score(self):
        obs = self._run_full_episode()
        assert obs.tool_result["terminal_breakdown"]["cabin_score"] == pytest.approx(1.0)

    def test_state_is_complete(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="medium")
        for pid, seat in FULL_ASSIGNMENTS_MEDIUM.items():
            assign(env, pid, seat)
        assert env.state.is_complete is True

    def test_step_after_done_raises(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="medium")
        for pid, seat in FULL_ASSIGNMENTS_MEDIUM.items():
            assign(env, pid, seat)
        with pytest.raises(RuntimeError, match="terminated"):
            assign(env, "PAX-001", "1A")


# ---------------------------------------------------------------------------
# 7. Step limit — episode terminates at max_steps (medium task)
# ---------------------------------------------------------------------------

class TestStepLimit:
    def test_episode_terminates_at_max_steps(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="medium")
        obs = None
        for _ in range(60):
            obs = fetch(env, "1A")
        assert obs.done is True
        assert obs.step_count == 60

    def test_incomplete_penalty_applied(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="medium")
        for _ in range(60):
            fetch(env, "1A")
        s = env.state
        assert s.passengers_assigned == 0
        assert s.is_complete is True

    def test_timeout_reason_in_observation(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="medium")
        obs = None
        for _ in range(60):
            obs = fetch(env, "1A")
        assert "timed out" in obs.reward_reason.lower()


# ---------------------------------------------------------------------------
# 8. Easy task
# ---------------------------------------------------------------------------

# Valid assignments for easy task — all 8 passengers, cabin class respected
FULL_ASSIGNMENTS_EASY = {
    # Business passengers → AC-2 business seats (1A–1D)
    "PAX-E001": "1A",
    "PAX-E002": "1B",
    "PAX-E003": "1C",
    "PAX-E004": "1D",
    # Economy passengers → AC-2 economy seats (2A–2F)
    "PAX-E005": "2A",
    "PAX-E006": "2B",
    "PAX-E007": "2C",
    "PAX-E008": "2D",
}


class TestEasyTask:
    def test_reset_easy_basic_fields(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="easy")

        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.passengers_remaining == 8
        assert obs.passengers_total == 8
        assert obs.tool_result is None
        assert obs.reward_reason == "Episode started"
        assert obs.step_count == 0

    def test_easy_max_steps_is_24(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="easy")
        assert obs.max_steps == 24  # 3 × 8

    def test_easy_ac1_has_8_occupied_seats(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="easy")
        assert len(obs.ac1_seats_occupied) == 8

    def test_easy_ac2_starts_empty_with_10_seats(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="easy")
        assert obs.ac2_seat_assignments == {}
        assert len(obs.ac2_seats_available) == 10  # AC-2 easy has 10 seats

    def test_easy_layouts_present(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="easy")
        assert obs.ac1_layout.get("aircraft_id") == "AC-1"
        assert obs.ac2_layout.get("aircraft_id") == "AC-2"

    def test_easy_fetch_returns_no_paid_window(self):
        env = make_env(task_id="easy")
        obs = fetch(env, "1A")
        assert obs.tool_result["status"] == "success"
        assert obs.tool_result["passenger_id"] == "PAX-E001"
        assert obs.tool_result["cabin"] == "business"
        assert obs.tool_result["paid_window"] is False

    def test_easy_assign_gives_no_preference_reason(self):
        """Easy passengers have no paid prefs — both satisfied fields should be None."""
        env = make_env(task_id="easy")
        obs = assign(env, "PAX-E001", "1A")
        assert obs.tool_result["status"] == "success"
        assert obs.tool_result["window_preference_satisfied"] is None
        assert obs.tool_result["legroom_preference_satisfied"] is None
        assert "preference" not in obs.reward_reason.lower() or "no preferences" in obs.reward_reason.lower()

    def test_easy_cabin_mismatch_penalised(self):
        """Assigning a business passenger to an economy seat must give negative reward."""
        env = make_env(task_id="easy")
        obs = assign(env, "PAX-E001", "2A")  # PAX-E001 is business → economy seat
        assert obs.tool_result["cabin_match"] is False
        assert obs.reward < 0

    def test_easy_full_episode_completes(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="easy")
        obs = None
        for pid, seat in FULL_ASSIGNMENTS_EASY.items():
            obs = assign(env, pid, seat)
        assert obs.done is True
        assert obs.passengers_remaining == 0

    def test_easy_grader_score_cabin_only(self):
        """
        With no paid_window passengers, grader_score == cabin_score.
        A perfect cabin assignment → grader_score == 1.0.
        """
        env = SeatReassignmentEnvironment()
        env.reset(task_id="easy")
        for pid, seat in FULL_ASSIGNMENTS_EASY.items():
            obs = assign(env, pid, seat)

        assert "grader_score" in obs.tool_result
        assert obs.tool_result["grader_score"] == pytest.approx(1.0)

    def test_easy_grader_score_in_range(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="easy")
        for pid, seat in FULL_ASSIGNMENTS_EASY.items():
            obs = assign(env, pid, seat)
        score = obs.tool_result["grader_score"]
        assert 0.0 <= score <= 1.0

    def test_easy_step_limit_terminates(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="easy")
        obs = None
        for _ in range(24):
            obs = fetch(env, "1A")
        assert obs.done is True
        assert obs.step_count == 24

    def test_easy_state_is_complete_after_full_assignment(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="easy")
        for pid, seat in FULL_ASSIGNMENTS_EASY.items():
            assign(env, pid, seat)
        assert env.state.is_complete is True


# ---------------------------------------------------------------------------
# 9. Hard task
# ---------------------------------------------------------------------------

# Optimal assignment for hard task — achieves grader_score = 1.0
# Business paid_window+legroom (PAX-001) → 2A (window, legroom) ✓✓
# Business paid_legroom only (PAX-002) → 2B (legroom) ✓
# Business paid_legroom only (PAX-005) → 2C (legroom) ✓
# Business paid_window only (PAX-004) → 1D (window, no legroom) ✓
# Business paid_window only (PAX-008) → 1A (window, no legroom) ✓
# Business no prefs (PAX-003, PAX-006, PAX-007) → remaining business
# Economy paid_window+legroom (PAX-009) → 4A (window, legroom) ✓✓
# Economy paid_legroom only (PAX-010) → 4B (legroom) ✓
# Economy paid_legroom only (PAX-012) → 4H (window, legroom) ✓
# Economy paid_window only (PAX-014) → 3H (window) ✓
# Economy paid_window only (PAX-015) → 3A (window) ✓
# Economy no prefs → any remaining economy

FULL_ASSIGNMENTS_HARD = {
    # Business (8 passengers → 8 business seats on AC-2)
    "PAX-001": "2A",  # paid_window + paid_legroom → window + legroom ✓✓
    "PAX-002": "2B",  # paid_legroom → legroom ✓
    "PAX-003": "1B",  # no pref
    "PAX-004": "1D",  # paid_window → window ✓
    "PAX-005": "2C",  # paid_legroom → legroom ✓
    "PAX-006": "1C",  # no pref
    "PAX-007": "2D",  # no pref (legroom bonus, but not paid for)
    "PAX-008": "1A",  # paid_window → window ✓
    # Economy (12 passengers → 12 of 16 economy seats on AC-2)
    "PAX-009": "4A",  # paid_window + paid_legroom → window + legroom ✓✓
    "PAX-010": "4B",  # paid_legroom → legroom ✓
    "PAX-011": "3B",  # no pref
    "PAX-012": "4H",  # paid_legroom → window + legroom (4H is window+legroom) ✓
    "PAX-013": "3C",  # no pref
    "PAX-014": "3H",  # paid_window → window ✓
    "PAX-015": "3A",  # paid_window → window ✓
    "PAX-016": "3D",  # no pref
    "PAX-017": "3E",  # no pref
    "PAX-018": "3F",  # no pref
    "PAX-019": "3G",  # no pref
    "PAX-020": "4C",  # no pref
}


class TestHardTask:
    def test_reset_hard_basic_fields(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="hard")

        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.passengers_remaining == 20
        assert obs.passengers_total == 20
        assert obs.tool_result is None
        assert obs.reward_reason == "Episode started"
        assert obs.step_count == 0

    def test_hard_max_steps_is_60(self):
        """Hard task: 3 × 20 passengers = 60 max steps."""
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="hard")
        assert obs.max_steps == 60

    def test_hard_ac1_has_20_occupied_seats(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="hard")
        assert len(obs.ac1_seats_occupied) == 20

    def test_hard_ac2_starts_empty_with_24_seats(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="hard")
        assert obs.ac2_seat_assignments == {}
        assert len(obs.ac2_seats_available) == 24

    def test_hard_layouts_present(self):
        env = SeatReassignmentEnvironment()
        obs = env.reset(task_id="hard")
        assert obs.ac1_layout.get("aircraft_id") == "AC-1"
        assert obs.ac2_layout.get("aircraft_id") == "AC-2"

    def test_hard_fetch_returns_paid_legroom(self):
        """PAX-001 is in 1A — has both paid_window and paid_legroom."""
        env = make_env(task_id="hard")
        obs = fetch(env, "1A")
        assert obs.tool_result["status"] == "success"
        assert obs.tool_result["passenger_id"] == "PAX-001"
        assert obs.tool_result["paid_window"] is True
        assert obs.tool_result["paid_legroom"] is True
        assert "current_seat_extra_legroom" in obs.tool_result  # AC-1 row 1 has legroom
        assert obs.tool_result["current_seat_extra_legroom"] is True

    def test_hard_fetch_no_prefs(self):
        """PAX-003 in 1C has no paid prefs."""
        env = make_env(task_id="hard")
        obs = fetch(env, "1C")
        assert obs.tool_result["paid_window"] is False
        assert obs.tool_result["paid_legroom"] is False

    def test_hard_assign_returns_both_preference_fields(self):
        """tool_assign_seat must return both window and legroom fields."""
        env = make_env(task_id="hard")
        obs = assign(env, "PAX-001", "2A")  # window+legroom seat
        assert obs.tool_result["status"] == "success"
        assert "window_preference_satisfied" in obs.tool_result
        assert "legroom_preference_satisfied" in obs.tool_result
        assert obs.tool_result["window_preference_satisfied"] is True   # 2A is window
        assert obs.tool_result["legroom_preference_satisfied"] is True  # 2A has legroom

    def test_hard_assign_window_pref_missed(self):
        """PAX-001 (paid_window) assigned to aisle seat → window_preference_satisfied=False."""
        env = make_env(task_id="hard")
        obs = assign(env, "PAX-001", "2B")  # 2B is aisle+legroom
        assert obs.tool_result["window_preference_satisfied"] is False
        assert obs.tool_result["legroom_preference_satisfied"] is True

    def test_hard_assign_legroom_pref_missed(self):
        """PAX-001 (paid_legroom) assigned to window seat without legroom → legroom_pref=False."""
        env = make_env(task_id="hard")
        obs = assign(env, "PAX-001", "1A")  # 1A is window, no legroom
        assert obs.tool_result["window_preference_satisfied"] is True
        assert obs.tool_result["legroom_preference_satisfied"] is False

    def test_hard_no_pref_passenger_both_fields_none(self):
        """PAX-003 has no prefs → both window and legroom should be None."""
        env = make_env(task_id="hard")
        obs = assign(env, "PAX-003", "1A")
        assert obs.tool_result["window_preference_satisfied"] is None
        assert obs.tool_result["legroom_preference_satisfied"] is None

    def test_hard_all_prefs_satisfied_gives_pref_reward(self):
        """PAX-001 → 2A (window+legroom) → REWARD_ASSIGN_CABIN_PREF."""
        from server.rewards import REWARD_ASSIGN_CABIN_PREF
        env = make_env(task_id="hard")
        obs = assign(env, "PAX-001", "2A")
        assert obs.reward == pytest.approx(REWARD_ASSIGN_CABIN_PREF)

    def test_hard_partial_prefs_gives_nopref_reward(self):
        """PAX-001 → 2B (legroom but not window) → REWARD_ASSIGN_CABIN_NOPREF (mixed)."""
        from server.rewards import REWARD_ASSIGN_CABIN_NOPREF
        env = make_env(task_id="hard")
        obs = assign(env, "PAX-001", "2B")
        assert obs.reward == pytest.approx(REWARD_ASSIGN_CABIN_NOPREF)

    def test_hard_cabin_mismatch_penalised(self):
        """Assigning a business passenger to economy seat → negative reward."""
        env = make_env(task_id="hard")
        obs = assign(env, "PAX-001", "3A")  # PAX-001 is business → economy seat
        assert obs.tool_result["cabin_match"] is False
        assert obs.reward < 0

    def test_hard_full_episode_completes(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="hard")
        obs = None
        for pid, seat in FULL_ASSIGNMENTS_HARD.items():
            obs = assign(env, pid, seat)
        assert obs.done is True
        assert obs.passengers_remaining == 0

    def test_hard_grader_score_perfect(self):
        """Optimal assignment satisfying all 12 constraints → grader_score = 1.0."""
        env = SeatReassignmentEnvironment()
        env.reset(task_id="hard")
        for pid, seat in FULL_ASSIGNMENTS_HARD.items():
            obs = assign(env, pid, seat)

        assert "grader_score" in obs.tool_result
        assert obs.tool_result["grader_score"] == pytest.approx(1.0)

    def test_hard_grader_score_in_range(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="hard")
        for pid, seat in FULL_ASSIGNMENTS_HARD.items():
            obs = assign(env, pid, seat)
        score = obs.tool_result["grader_score"]
        assert 0.0 <= score <= 1.0

    def test_hard_step_limit_terminates_at_60(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="hard")
        obs = None
        for _ in range(60):
            obs = fetch(env, "1A")
        assert obs.done is True
        assert obs.step_count == 60

    def test_hard_state_is_complete_after_full_assignment(self):
        env = SeatReassignmentEnvironment()
        env.reset(task_id="hard")
        for pid, seat in FULL_ASSIGNMENTS_HARD.items():
            assign(env, pid, seat)
        assert env.state.is_complete is True
