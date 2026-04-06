"""
Tests for server/rewards.py

All DataFrames and dicts are constructed inline — no file I/O.
Run with: pytest tests/test_rewards.py -v
"""

import pandas as pd
import pytest

from server.rewards import (
    RewardComputer,
    REWARD_FETCH_NEUTRAL,
    REWARD_FETCH_REDUNDANT,
    REWARD_FETCH_ERROR,
    REWARD_ASSIGN_CABIN_PREF,
    REWARD_ASSIGN_CABIN_NOPREF,
    REWARD_ASSIGN_CABIN_PREFMISS,
    REWARD_ASSIGN_CABIN_MISMATCH,
    REWARD_ASSIGN_ERROR,
    REWARD_SWAP_IMPROVE,
    REWARD_SWAP_NEUTRAL,
    REWARD_SWAP_WORSE,
    REWARD_SWAP_ERROR,
    REWARD_INVALID_TOOL,
    REWARD_INCOMPLETE_PENALTY,
    TERMINAL_W_CABIN,
    TERMINAL_W_PREF,
    TERMINAL_W_EFF,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rc():
    """RewardComputer with 4 total passengers and 20 max steps."""
    return RewardComputer(total_passengers=4, max_steps=20)


# Minimal passengers: P1 (business, paid_window), P2 (business, no pref),
#                     P3 (economy, no pref),      P4 (economy, no pref)
PASSENGERS = pd.DataFrame([
    {"passenger_id": "P1", "name": "Alice", "seat_ac1": "1A", "cabin": "business", "paid_window": True},
    {"passenger_id": "P2", "name": "Bob",   "seat_ac1": "1B", "cabin": "business", "paid_window": False},
    {"passenger_id": "P3", "name": "Carol", "seat_ac1": "2A", "cabin": "economy",  "paid_window": False},
    {"passenger_id": "P4", "name": "Dave",  "seat_ac1": "2B", "cabin": "economy",  "paid_window": False},
])

# AC-2 seat info used across terminal/grader tests
AC2 = {
    "B1": {"cabin": "business", "seat_type": "window"},
    "B2": {"cabin": "business", "seat_type": "aisle"},
    "E1": {"cabin": "economy",  "seat_type": "window"},
    "E2": {"cabin": "economy",  "seat_type": "aisle"},
}


def _assignments(mapping: dict, include_ac1: bool = True) -> pd.DataFrame:
    """
    Build an assignments DataFrame from {passenger_id: seat_ac2_or_None}.
    Set index to passenger_id, mirroring what the environment does at reset().
    """
    rows = []
    for pid, seat_ac2 in mapping.items():
        row = {"passenger_id": pid, "seat_ac2": seat_ac2}
        if include_ac1:
            # seat_ac1 not used by reward methods but keeps the schema consistent
            row["seat_ac1"] = pid.replace("P", "S")
        rows.append(row)
    return pd.DataFrame(rows).set_index("passenger_id")


# ===========================================================================
# Fetch rewards
# ===========================================================================

class TestRewardForFetch:
    def test_neutral_new_seat(self, rc):
        reward, reason = rc.reward_for_fetch(is_redundant=False, is_error=False)
        assert reward == REWARD_FETCH_NEUTRAL
        assert "new seat" in reason.lower() or "fetch" in reason.lower()

    def test_redundant_seat(self, rc):
        reward, reason = rc.reward_for_fetch(is_redundant=True, is_error=False)
        assert reward == REWARD_FETCH_REDUNDANT
        assert "redundant" in reason.lower()

    def test_error_seat(self, rc):
        reward, reason = rc.reward_for_fetch(is_redundant=False, is_error=True)
        assert reward == REWARD_FETCH_ERROR
        assert "error" in reason.lower()

    def test_error_takes_precedence_over_redundant(self, rc):
        """If a seat was already fetched AND the tool errors, error penalty applies."""
        reward, _ = rc.reward_for_fetch(is_redundant=True, is_error=True)
        assert reward == REWARD_FETCH_ERROR


# ===========================================================================
# Assign rewards
# ===========================================================================

class TestRewardForAssign:
    def _result(self, cabin_match, preference_satisfied, status="success"):
        return {"status": status, "cabin_match": cabin_match,
                "preference_satisfied": preference_satisfied}

    def test_cabin_match_preference_satisfied(self, rc):
        reward, reason = rc.reward_for_assign(
            self._result(cabin_match=True, preference_satisfied=True))
        assert reward == REWARD_ASSIGN_CABIN_PREF
        assert "preference satisfied" in reason.lower()

    def test_cabin_match_no_preference(self, rc):
        reward, reason = rc.reward_for_assign(
            self._result(cabin_match=True, preference_satisfied=None))
        assert reward == REWARD_ASSIGN_CABIN_NOPREF
        assert "no window preference" in reason.lower()

    def test_cabin_match_preference_missed(self, rc):
        reward, reason = rc.reward_for_assign(
            self._result(cabin_match=True, preference_satisfied=False))
        assert reward == REWARD_ASSIGN_CABIN_PREFMISS
        assert "not satisfied" in reason.lower()

    def test_cabin_mismatch(self, rc):
        reward, reason = rc.reward_for_assign(
            self._result(cabin_match=False, preference_satisfied=None))
        assert reward == REWARD_ASSIGN_CABIN_MISMATCH
        assert "mismatch" in reason.lower()

    def test_cabin_mismatch_ignores_preference(self, rc):
        """Cabin mismatch penalty applies even if preference_satisfied happens to be True."""
        reward, _ = rc.reward_for_assign(
            self._result(cabin_match=False, preference_satisfied=True))
        assert reward == REWARD_ASSIGN_CABIN_MISMATCH

    def test_error_result(self, rc):
        result = {"status": "error", "message": "Seat does not exist"}
        reward, reason = rc.reward_for_assign(result)
        assert reward == REWARD_ASSIGN_ERROR
        assert "error" in reason.lower()


# ===========================================================================
# Swap rewards
# ===========================================================================

class TestRewardForSwap:
    """
    Passengers used:
      pax_biz_pref  — business, paid_window=True
      pax_biz_nopref — business, paid_window=False
    Seats used:
      biz_window — business window
      biz_aisle  — business aisle
    """

    PAX_PREF   = {"cabin": "business", "paid_window": True}
    PAX_NOPREF = {"cabin": "business", "paid_window": False}
    BIZ_WIN    = {"cabin": "business", "seat_type": "window"}
    BIZ_AISLE  = {"cabin": "business", "seat_type": "aisle"}

    def _swap(self, rc, pax1_seat, pax2_seat, pax1_new, pax2_new, pax1=None, pax2=None):
        pax1 = pax1 or self.PAX_PREF
        pax2 = pax2 or self.PAX_NOPREF
        return rc.reward_for_swap(
            tool_result={"status": "success"},
            pax1_info=pax1, pax2_info=pax2,
            old_seat_1_info=pax1_seat, old_seat_2_info=pax2_seat,
            new_seat_1_info=pax1_new,  new_seat_2_info=pax2_new,
        )

    def test_improvement(self, rc):
        # pax_pref in aisle → window: constraint +1.5 (aisle→window for paid_window)
        reward, reason = self._swap(
            rc,
            pax1_seat=self.BIZ_AISLE, pax2_seat=self.BIZ_WIN,   # before
            pax1_new=self.BIZ_WIN,    pax2_new=self.BIZ_AISLE,   # after
        )
        assert reward == REWARD_SWAP_IMPROVE
        assert "improved" in reason.lower()

    def test_worse(self, rc):
        # pax_pref in window → aisle: constraint drops
        reward, reason = self._swap(
            rc,
            pax1_seat=self.BIZ_WIN,   pax2_seat=self.BIZ_AISLE,
            pax1_new=self.BIZ_AISLE,  pax2_new=self.BIZ_WIN,
        )
        assert reward == REWARD_SWAP_WORSE
        assert "worsened" in reason.lower()

    def test_neutral(self, rc):
        # Neither passenger has paid_window; both stay in same cabin — scores identical.
        pax = {"cabin": "economy", "paid_window": False}
        eco_win   = {"cabin": "economy", "seat_type": "window"}
        eco_aisle = {"cabin": "economy", "seat_type": "aisle"}
        reward, reason = rc.reward_for_swap(
            tool_result={"status": "success"},
            pax1_info=pax, pax2_info=pax,
            old_seat_1_info=eco_win,   old_seat_2_info=eco_aisle,
            new_seat_1_info=eco_aisle, new_seat_2_info=eco_win,
        )
        assert reward == REWARD_SWAP_NEUTRAL
        assert "did not change" in reason.lower()

    def test_error(self, rc):
        result = {"status": "error", "message": "Passenger not yet assigned to AC-2"}
        reward, reason = rc.reward_for_swap(
            tool_result=result,
            pax1_info=self.PAX_PREF,  pax2_info=self.PAX_NOPREF,
            old_seat_1_info=self.BIZ_WIN,   old_seat_2_info=self.BIZ_AISLE,
            new_seat_1_info=self.BIZ_AISLE, new_seat_2_info=self.BIZ_WIN,
        )
        assert reward == REWARD_SWAP_ERROR
        assert "error" in reason.lower()


# ===========================================================================
# Invalid tool
# ===========================================================================

def test_reward_for_invalid_tool(rc):
    reward, reason = rc.reward_for_invalid_tool()
    assert reward == REWARD_INVALID_TOOL
    assert "unrecognized" in reason.lower()


# ===========================================================================
# Terminal reward
# ===========================================================================

class TestTerminalReward:
    def test_perfect_assignment(self, rc):
        """All cabin matches, paid-window passenger in window, low step count."""
        asgn = _assignments({"P1": "B1", "P2": "B2", "P3": "E1", "P4": "E2"})
        reward, bd = rc.terminal_reward(asgn, PASSENGERS, AC2, total_steps=4)

        # cabin_score = 4/4 = 1.0
        assert bd["cabin_score"] == pytest.approx(1.0)
        # P1 (paid_window) → B1 (business window) → satisfied
        assert bd["preference_score"] == pytest.approx(1.0)
        # efficiency = 1 - 4/20 = 0.8
        assert bd["efficiency_score"] == pytest.approx(0.8)
        assert bd["incomplete_penalty"] == 0.0
        expected = TERMINAL_W_CABIN * 1.0 + TERMINAL_W_PREF * 1.0 + TERMINAL_W_EFF * 0.8
        assert reward == pytest.approx(expected)

    def test_all_wrong_assignment(self, rc):
        """All in wrong cabin, paid-window pax not in window, max steps used."""
        # Business passengers → economy seats; economy → business
        # P1 (paid_window, business) → E2 (economy aisle): cabin wrong, pref unsatisfied
        asgn = _assignments({"P1": "E2", "P2": "E1", "P3": "B2", "P4": "B1"})
        reward, bd = rc.terminal_reward(asgn, PASSENGERS, AC2, total_steps=20)

        assert bd["cabin_score"] == pytest.approx(0.0)
        assert bd["preference_score"] == pytest.approx(0.0)   # P1 → E2 (aisle, not window)
        assert bd["efficiency_score"] == pytest.approx(0.0)   # 1 - 20/20
        assert bd["incomplete_penalty"] == 0.0
        assert reward == pytest.approx(0.0)

    def test_partial_assignment(self, rc):
        """Only 2 of 4 passengers assigned — incomplete penalty fires."""
        asgn = _assignments({"P1": "B1", "P2": "B2", "P3": None, "P4": None})
        reward, bd = rc.terminal_reward(asgn, PASSENGERS, AC2, total_steps=10)

        # cabin_score = 2 correct / 4 total = 0.5
        assert bd["cabin_score"] == pytest.approx(0.5)
        # P1 assigned to B1 (window) → pref satisfied; 1/1
        assert bd["preference_score"] == pytest.approx(1.0)
        assert bd["incomplete_penalty"] == REWARD_INCOMPLETE_PENALTY
        assert "incomplete_penalty" in bd

    def test_incomplete_at_step_limit(self, rc):
        """Episode ends at max_steps with passengers unassigned — penalty applied."""
        asgn = _assignments({"P1": None, "P2": None, "P3": None, "P4": None})
        reward, bd = rc.terminal_reward(asgn, PASSENGERS, AC2, total_steps=20)

        assert bd["cabin_score"] == pytest.approx(0.0)
        assert bd["incomplete_penalty"] == REWARD_INCOMPLETE_PENALTY
        assert bd["efficiency_score"] == pytest.approx(0.0)

    def test_breakdown_keys_present(self, rc):
        asgn = _assignments({"P1": "B1", "P2": "B2", "P3": "E1", "P4": "E2"})
        _, bd = rc.terminal_reward(asgn, PASSENGERS, AC2, total_steps=8)
        assert set(bd.keys()) == {
            "cabin_score", "preference_score", "efficiency_score",
            "weighted_total", "incomplete_penalty",
        }

    def test_weighted_total_matches_return_value(self, rc):
        asgn = _assignments({"P1": "B1", "P2": "B2", "P3": "E1", "P4": "E2"})
        reward, bd = rc.terminal_reward(asgn, PASSENGERS, AC2, total_steps=8)
        assert reward == pytest.approx(bd["weighted_total"])


# ===========================================================================
# Grader score
# ===========================================================================

class TestGraderScore:
    def test_perfect_returns_one(self, rc):
        asgn = _assignments({"P1": "B1", "P2": "B2", "P3": "E1", "P4": "E2"})
        score = rc.grader_score(asgn, PASSENGERS, AC2)
        assert score == pytest.approx(1.0)

    def test_all_wrong_returns_zero(self, rc):
        # cabin_score = 0, P1 (paid_window) ends up in economy aisle → pref = 0
        asgn = _assignments({"P1": "E2", "P2": "E1", "P3": "B2", "P4": "B1"})
        score = rc.grader_score(asgn, PASSENGERS, AC2)
        assert score == pytest.approx(0.0)

    def test_partial_is_between_zero_and_one(self, rc):
        # 2 correctly placed (including paid-window pax), 2 unassigned
        asgn = _assignments({"P1": "B1", "P2": "B2", "P3": None, "P4": None})
        score = rc.grader_score(asgn, PASSENGERS, AC2)
        assert 0.0 < score < 1.0

    def test_no_pref_passengers_equals_cabin_score(self):
        """When no passenger has paid_window, grader_score == cabin fraction."""
        rc_nopref = RewardComputer(total_passengers=2, max_steps=10)
        pax = pd.DataFrame([
            {"passenger_id": "X1", "name": "A", "seat_ac1": "1A", "cabin": "economy", "paid_window": False},
            {"passenger_id": "X2", "name": "B", "seat_ac1": "1B", "cabin": "economy", "paid_window": False},
        ])
        asgn = _assignments({"X1": "E1", "X2": "E2"})
        score = rc_nopref.grader_score(asgn, pax, AC2)
        # both economy → economy: cabin_score = 1.0, no pref → returns cabin_score
        assert score == pytest.approx(1.0)

    def test_unassigned_pref_passenger_penalises_score(self, rc):
        """A paid-window pax left unassigned should lower score vs all assigned."""
        asgn_full    = _assignments({"P1": "B1", "P2": "B2", "P3": "E1", "P4": "E2"})
        asgn_partial = _assignments({"P1": None, "P2": "B2", "P3": "E1", "P4": "E2"})
        score_full    = rc.grader_score(asgn_full,    PASSENGERS, AC2)
        score_partial = rc.grader_score(asgn_partial, PASSENGERS, AC2)
        assert score_partial < score_full


# ===========================================================================
# _constraint_score (internal helper, tested directly for clarity)
# ===========================================================================

class TestConstraintScore:
    def setup_method(self):
        self.rc = RewardComputer(total_passengers=20, max_steps=60)

    def test_correct_cabin_window_pref_satisfied(self):
        pax  = {"cabin": "business", "paid_window": True}
        seat = {"cabin": "business", "seat_type": "window"}
        assert self.rc._constraint_score(pax, seat) == pytest.approx(2.0)

    def test_correct_cabin_window_pref_missed(self):
        pax  = {"cabin": "business", "paid_window": True}
        seat = {"cabin": "business", "seat_type": "aisle"}
        assert self.rc._constraint_score(pax, seat) == pytest.approx(0.5)

    def test_correct_cabin_no_pref(self):
        pax  = {"cabin": "economy", "paid_window": False}
        seat = {"cabin": "economy", "seat_type": "middle"}
        assert self.rc._constraint_score(pax, seat) == pytest.approx(1.0)

    def test_wrong_cabin_no_pref(self):
        pax  = {"cabin": "business", "paid_window": False}
        seat = {"cabin": "economy", "seat_type": "aisle"}
        assert self.rc._constraint_score(pax, seat) == pytest.approx(0.0)

    def test_wrong_cabin_pref_missed(self):
        pax  = {"cabin": "business", "paid_window": True}
        seat = {"cabin": "economy", "seat_type": "aisle"}
        assert self.rc._constraint_score(pax, seat) == pytest.approx(-0.5)
