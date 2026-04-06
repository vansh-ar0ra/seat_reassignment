"""
Integration tests for AirlineReassignmentEnvironment.

Tests instantiate the environment directly — no server, no WebSocket.
All 20 passengers and their AC-1 seat assignments are fixed by the data
generation seed, so the expected values here are deterministic.

AC-1 seat → passenger mapping (from data/passengers.csv, seed=42):
  Business (rows 1-2):  PAX-001→1A, PAX-002→1B, PAX-003→1C, PAX-004→1D
                        PAX-005→2A, PAX-006→2B, PAX-007→2C, PAX-008→2D
  Economy  (rows 3-4):  PAX-009→3A, PAX-010→3B, PAX-011→3C, PAX-012→3D
                        PAX-013→3E, PAX-014→3F, PAX-015→4A, PAX-016→4B
                        PAX-017→4C, PAX-018→4D, PAX-019→4E, PAX-020→4F

Run with: pytest tests/test_environment.py -v
"""

import pytest

from models import AirlineReassignmentAction, AirlineReassignmentObservation
from server.environment import AirlineReassignmentEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env() -> AirlineReassignmentEnvironment:
    env = AirlineReassignmentEnvironment()
    env.reset()
    return env


def fetch(env, seat_id: str) -> AirlineReassignmentObservation:
    return env.step(AirlineReassignmentAction(tool_name="get_passenger_details", args={"seat_id": seat_id}))


def assign(env, passenger_id: str, target_seat_id: str) -> AirlineReassignmentObservation:
    return env.step(AirlineReassignmentAction(
        tool_name="assign_seat",
        args={"passenger_id": passenger_id, "target_seat_id": target_seat_id},
    ))


def swap(env, pid1: str, pid2: str) -> AirlineReassignmentObservation:
    return env.step(AirlineReassignmentAction(
        tool_name="swap_seats",
        args={"passenger_id_1": pid1, "passenger_id_2": pid2},
    ))


# ---------------------------------------------------------------------------
# 1. Reset produces a valid initial observation
# ---------------------------------------------------------------------------

class TestReset:
    def test_basic_fields(self):
        env = AirlineReassignmentEnvironment()
        obs = env.reset()

        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.passengers_remaining == 20
        assert obs.passengers_total == 20
        assert obs.tool_result is None
        assert obs.reward_reason == "Episode started"
        assert obs.step_count == 0

    def test_all_ac1_seats_occupied(self):
        env = AirlineReassignmentEnvironment()
        obs = env.reset()
        assert len(obs.ac1_seats_occupied) == 20

    def test_ac2_starts_empty(self):
        env = AirlineReassignmentEnvironment()
        obs = env.reset()
        assert obs.ac2_seat_assignments == {}
        assert len(obs.ac2_seats_available) == 24  # AC-2 has 24 seats

    def test_layouts_present(self):
        env = AirlineReassignmentEnvironment()
        obs = env.reset()
        assert obs.ac1_layout.get("aircraft_id") == "AC-1"
        assert obs.ac2_layout.get("aircraft_id") == "AC-2"

    def test_max_steps_is_three_times_passengers(self):
        env = AirlineReassignmentEnvironment()
        obs = env.reset()
        assert obs.max_steps == 60  # 3 × 20


# ---------------------------------------------------------------------------
# 2. Fetch then assign — happy path
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

    def test_grader_score_none_after_reset(self):
        env = AirlineReassignmentEnvironment()
        obs = env.reset()
        assert obs.grader_score is None


# ---------------------------------------------------------------------------
# 3. Invalid tool name
# ---------------------------------------------------------------------------

class TestInvalidTool:
    def test_unknown_tool_returns_error(self):
        env = make_env()
        obs = env.step(AirlineReassignmentAction(tool_name="teleport_passenger", args={}))

        assert obs.tool_result["status"] == "error"
        assert "teleport_passenger" in obs.tool_result["message"]

    def test_unknown_tool_gives_negative_reward(self):
        env = make_env()
        obs = env.step(AirlineReassignmentAction(tool_name="nonexistent", args={}))
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
        obs = env.step(AirlineReassignmentAction(tool_name="get_passenger_details", args={"seat_id": "1A"}))
        # After swap: PAX-001 should be in 1D, PAX-004 in 1B
        # Reading observation ac2_seat_assignments
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
# 6. Episode completion — all 20 passengers assigned
# ---------------------------------------------------------------------------

# Known valid assignments that respect cabin constraints
FULL_ASSIGNMENTS = {
    # Business passengers → AC-2 business seats
    "PAX-001": "1A", "PAX-002": "1B", "PAX-003": "1C", "PAX-004": "1D",
    "PAX-005": "2A", "PAX-006": "2B", "PAX-007": "2C", "PAX-008": "2D",
    # Economy passengers → AC-2 economy seats
    "PAX-009": "3A", "PAX-010": "3B", "PAX-011": "3C", "PAX-012": "3D",
    "PAX-013": "3E", "PAX-014": "3F", "PAX-015": "4A", "PAX-016": "4B",
    "PAX-017": "4C", "PAX-018": "4D", "PAX-019": "4E", "PAX-020": "4F",
}


class TestEpisodeCompletion:
    def _run_full_episode(self) -> AirlineReassignmentObservation:
        env = AirlineReassignmentEnvironment()
        env.reset()
        obs = None
        for pid, seat in FULL_ASSIGNMENTS.items():
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
        # Last step reward = step reward + terminal_reward; terminal should be positive
        assert obs.reward > 0

    def test_grader_score_in_observation(self):
        obs = self._run_full_episode()
        assert obs.grader_score is not None
        assert 0.0 <= obs.grader_score <= 1.0

    def test_grader_score_present_mid_episode(self):
        env = AirlineReassignmentEnvironment()
        env.reset()
        obs = assign(env, "PAX-001", "1A")
        # Grader score should be live after any step, not just at episode end
        assert obs.grader_score is not None
        assert 0.0 <= obs.grader_score <= 1.0

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
        env = AirlineReassignmentEnvironment()
        env.reset()
        for pid, seat in FULL_ASSIGNMENTS.items():
            assign(env, pid, seat)
        assert env.state.is_complete is True

    def test_step_after_done_raises(self):
        env = AirlineReassignmentEnvironment()
        env.reset()
        for pid, seat in FULL_ASSIGNMENTS.items():
            assign(env, pid, seat)
        with pytest.raises(RuntimeError, match="terminated"):
            assign(env, "PAX-001", "1A")


# ---------------------------------------------------------------------------
# 7. Step limit — episode terminates at max_steps
# ---------------------------------------------------------------------------

class TestStepLimit:
    def test_episode_terminates_at_max_steps(self):
        env = AirlineReassignmentEnvironment()
        env.reset()
        obs = None
        # Fetch seat 1A on every step — redundant after the first
        for _ in range(60):
            obs = fetch(env, "1A")
        assert obs.done is True
        assert obs.step_count == 60

    def test_incomplete_penalty_applied(self):
        env = AirlineReassignmentEnvironment()
        env.reset()
        for _ in range(60):
            fetch(env, "1A")
        # State should reflect 0 assignments
        s = env.state
        assert s.passengers_assigned == 0
        assert s.is_complete is True  # done=True even though incomplete

    def test_timeout_reason_in_observation(self):
        env = AirlineReassignmentEnvironment()
        env.reset()
        obs = None
        for _ in range(60):
            obs = fetch(env, "1A")
        assert "timed out" in obs.reward_reason.lower()
