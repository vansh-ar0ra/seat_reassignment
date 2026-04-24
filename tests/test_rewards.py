"""
Tests for server/rewards.py

All data is constructed inline — no file I/O.
Run with: pytest tests/test_rewards.py -v
"""

import pytest

from server.rewards import (
    RewardComputer,
    EPS,
    PRIORITY_WEIGHTS,
    priority_weight,
    REWARD_LIST_PASSENGERS_FIRST,
    REWARD_LIST_PASSENGERS_CHURN,
    REWARD_GET_DETAILS_UNBOOKED,
    REWARD_GET_DETAILS_BOOKED,
    REWARD_LIST_FLIGHTS,
    REWARD_GET_FLIGHT_DETAILS,
    REWARD_SAME_CABIN_GROUP,
    REWARD_UPGRADE,
    REWARD_DOWNGRADE,
    REWARD_DEADLINE_BONUS,
    REWARD_HARD_VIOLATION,
    REWARD_FAILED_BOOKING,
    REWARD_INVALID_TOOL,
    GRADER_W_COVERAGE,
    GRADER_W_CABIN_MATCH,
    GRADER_W_GROUP_INTEGRITY,
    GRADER_W_DEADLINE,
    GRADER_W_SSR_INTEGRITY,
    GRADER_HARD_PENALTY,
    GROUP_SAME_FLIGHT_SAME_CABIN,
    GROUP_SAME_FLIGHT_DIFF_CABIN,
    GROUP_SPLIT_FLIGHTS_HARD,
    GROUP_SPLIT_FLIGHTS_SOFT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rc():
    """RewardComputer with 10 passengers and 60 max steps."""
    return RewardComputer(total_passengers=10, max_steps=60)


# Reusable test data — passengers, flights, bookings
PASSENGERS = {
    "P1": {"passenger_id": "P1", "name": "A", "priority_tier": 1, "original_cabin": "business",
           "group_id": None, "group_integrity": None, "group_size": None,
           "ssr_flags": ["UM"], "downstream_deadline": "14:00"},
    "P2": {"passenger_id": "P2", "name": "B", "priority_tier": 2, "original_cabin": "business",
           "group_id": "G1", "group_integrity": "hard", "group_size": 2,
           "ssr_flags": [], "downstream_deadline": None},
    "P3": {"passenger_id": "P3", "name": "C", "priority_tier": 3, "original_cabin": "economy",
           "group_id": "G1", "group_integrity": "hard", "group_size": 2,
           "ssr_flags": [], "downstream_deadline": None},
    "P4": {"passenger_id": "P4", "name": "D", "priority_tier": 4, "original_cabin": "economy",
           "group_id": "G2", "group_integrity": "soft", "group_size": 2,
           "ssr_flags": [], "downstream_deadline": None},
    "P5": {"passenger_id": "P5", "name": "E", "priority_tier": 5, "original_cabin": "economy",
           "group_id": "G2", "group_integrity": "soft", "group_size": 2,
           "ssr_flags": ["WCHR"], "downstream_deadline": "16:00"},
}

GROUPS = {
    "G1": ["P2", "P3"],
    "G2": ["P4", "P5"],
}

FLIGHTS = {
    "FL-A": {"flight_id": "FL-A", "departure_time": "09:00", "arrival_time": "12:00",
             "cabin_availability": {"economy": 5, "business": 3},
             "supports_ssr": ["UM", "WCHR"]},
    "FL-B": {"flight_id": "FL-B", "departure_time": "13:00", "arrival_time": "16:00",
             "cabin_availability": {"economy": 3, "business": 2},
             "supports_ssr": ["WCHR"]},
}


# ===========================================================================
# 1. TestInfoCallRewards
# ===========================================================================

class TestInfoCallRewards:
    def test_first_list_passengers_positive(self, rc):
        """First call to list_passengers should give a small positive reward."""
        # Simulate ep_state
        class FakeEp:
            info_calls = {"list_passengers": 1}
            last_booking_step = 0
            step_count = 1
        reward, reason = rc.reward_for_info_call("list_passengers", FakeEp())
        assert reward == REWARD_LIST_PASSENGERS_FIRST
        assert reward > 0

    def test_repeated_list_passengers_negative(self, rc):
        """5th+ call with no intervening bookings -> churn penalty."""
        class FakeEp:
            info_calls = {"list_passengers": 5}
            last_booking_step = 0
            step_count = 10
        reward, reason = rc.reward_for_info_call("list_passengers", FakeEp())
        assert reward == REWARD_LIST_PASSENGERS_CHURN
        assert reward < 0

    def test_get_details_unbooked_positive(self, rc):
        class FakeEp:
            pass
        reward, _ = rc.reward_for_info_call("get_passenger_details", FakeEp())
        assert reward == REWARD_GET_DETAILS_UNBOOKED
        assert reward > 0

    def test_get_details_booked_negative(self, rc):
        class FakeEp:
            pass
        reward, _ = rc.reward_for_info_call("get_passenger_details_booked", FakeEp())
        assert reward == REWARD_GET_DETAILS_BOOKED
        assert reward < 0

    def test_list_flights_always_small_positive(self, rc):
        class FakeEp:
            info_calls = {"list_alternative_flights": 3}
        reward, _ = rc.reward_for_info_call("list_alternative_flights", FakeEp())
        assert reward == REWARD_LIST_FLIGHTS
        assert reward > 0


# ===========================================================================
# 2. TestBookingRewards
# ===========================================================================

class TestBookingRewards:
    def test_same_cabin_high_reward(self, rc):
        pax = {"priority_tier": 1, "original_cabin": "business"}
        result = {"status": "success", "cabin": "business", "original_cabin": "business"}
        reward, _ = rc.reward_for_booking(result, pax, None)
        assert reward == pytest.approx(REWARD_SAME_CABIN_GROUP * priority_weight(1))

    def test_upgrade_medium_reward(self, rc):
        pax = {"priority_tier": 3, "original_cabin": "economy"}
        result = {"status": "success", "cabin": "business", "original_cabin": "economy"}
        reward, _ = rc.reward_for_booking(result, pax, None)
        assert reward == pytest.approx(REWARD_UPGRADE * priority_weight(3))

    def test_downgrade_small_reward(self, rc):
        pax = {"priority_tier": 2, "original_cabin": "business"}
        result = {"status": "success", "cabin": "economy", "original_cabin": "business"}
        reward, _ = rc.reward_for_booking(result, pax, None)
        assert reward == pytest.approx(REWARD_DOWNGRADE * priority_weight(2))

    def test_priority_weight_scaling(self, rc):
        """Tier 1 gets higher reward than Tier 5 for the same outcome."""
        result = {"status": "success", "cabin": "economy", "original_cabin": "economy"}
        pax1 = {"priority_tier": 1, "original_cabin": "economy"}
        pax5 = {"priority_tier": 5, "original_cabin": "economy"}
        r1, _ = rc.reward_for_booking(result, pax1, None)
        r5, _ = rc.reward_for_booking(result, pax5, None)
        assert r1 > r5

    def test_deadline_met_bonus(self, rc):
        pax = {"priority_tier": 2, "original_cabin": "economy"}
        result = {"status": "success", "cabin": "economy", "original_cabin": "economy",
                  "deadline_met": True}
        reward_with, _ = rc.reward_for_booking(result, pax, None)

        result_no_dl = {"status": "success", "cabin": "economy", "original_cabin": "economy"}
        reward_without, _ = rc.reward_for_booking(result_no_dl, pax, None)

        assert reward_with > reward_without
        expected_bonus = REWARD_DEADLINE_BONUS * priority_weight(2)
        assert reward_with == pytest.approx(reward_without + expected_bonus)

    def test_failed_booking_small_penalty(self, rc):
        result = {"status": "error", "message": "No seats available"}
        pax = {"priority_tier": 3, "original_cabin": "economy"}
        reward, _ = rc.reward_for_booking(result, pax, None)
        assert reward == REWARD_FAILED_BOOKING
        assert reward < 0

    def test_invalid_tool_penalty(self, rc):
        reward, _ = rc.reward_for_invalid_tool()
        assert reward == REWARD_INVALID_TOOL
        assert reward < 0


# ===========================================================================
# 3. TestGraderScore
# ===========================================================================

class TestGraderScore:
    def test_perfect_score_all_booked_same_cabin(self, rc):
        """All passengers booked in same cabin, no SSR violations, deadlines met."""
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business"},
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},
            "P4": {"flight_id": "FL-B", "cabin": "economy"},
            "P5": {"flight_id": "FL-B", "cabin": "economy"},
        }
        score = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        # G1 on same flight, G2 on same flight, SSR OK, deadlines OK
        # Note: G1 is hard group — P2 (business) and P3 (economy) same flight diff cabin -> 0.7
        # This won't be 1.0 because G1 has split cabin
        assert score > 0.9

    def test_zero_coverage(self, rc):
        """No passengers booked -> low score (only SSR integrity is 1.0 since no violations possible)."""
        bookings = {}
        score = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        # coverage=0, cabin_match=0, group=0, deadline=0, ssr=1.0 -> 0.20*1.0 = 0.20
        assert score == pytest.approx(GRADER_W_SSR_INTEGRITY * 1.0)

    def test_partial_coverage(self, rc):
        """Some passengers booked, some not."""
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business"},
            "P2": {"flight_id": "FL-A", "cabin": "business"},
        }
        score = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        assert EPS < score < 1.0 - EPS

    def test_ssr_violation_penalizes(self, rc):
        """Book P1 (UM) on FL-B which doesn't support UM -> SSR violation."""
        bookings = {
            "P1": {"flight_id": "FL-B", "cabin": "business"},  # UM not on FL-B
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        score_bad = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)

        bookings_good = dict(bookings)
        bookings_good["P1"] = {"flight_id": "FL-A", "cabin": "business"}
        score_good = rc.grader_score(bookings_good, PASSENGERS, FLIGHTS, GROUPS)

        assert score_bad < score_good

    def test_group_split_penalizes(self, rc):
        """Hard group G1 split across flights -> score drops."""
        bookings_split = {
            "P1": {"flight_id": "FL-A", "cabin": "business"},
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-B", "cabin": "economy"},  # G1 split!
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        bookings_together = {
            "P1": {"flight_id": "FL-A", "cabin": "business"},
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        score_split = rc.grader_score(bookings_split, PASSENGERS, FLIGHTS, GROUPS)
        score_together = rc.grader_score(bookings_together, PASSENGERS, FLIGHTS, GROUPS)
        assert score_split < score_together

    def test_deadline_missed_penalizes(self, rc):
        """P1 has deadline 14:00. FL-B arrives 16:00 -> deadline missed."""
        bookings_missed = {
            "P1": {"flight_id": "FL-B", "cabin": "business"},  # arrives 16:00, deadline 14:00
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        bookings_met = dict(bookings_missed)
        bookings_met["P1"] = {"flight_id": "FL-A", "cabin": "business"}
        score_missed = rc.grader_score(bookings_missed, PASSENGERS, FLIGHTS, GROUPS)
        score_met = rc.grader_score(bookings_met, PASSENGERS, FLIGHTS, GROUPS)
        assert score_missed < score_met

    def test_grader_is_deterministic(self, rc):
        """Same input -> same output."""
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business"},
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        s1 = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        s2 = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        assert s1 == s2

    def test_grader_clamped_to_eps_range(self, rc):
        """Score is always in (EPS, 1-EPS)."""
        # Zero coverage
        score_zero = rc.grader_score({}, PASSENGERS, FLIGHTS, GROUPS)
        assert score_zero >= EPS

        # Full coverage (best effort)
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business"},
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        score_full = rc.grader_score(bookings, PASSENGERS, FLIGHTS, GROUPS)
        assert score_full <= 1.0 - EPS


# ===========================================================================
# 4. TestPriorityWeights
# ===========================================================================

class TestPriorityWeights:
    def test_tier1_highest(self):
        assert priority_weight(1) == 1.5
        for tier in [2, 3, 4, 5]:
            assert priority_weight(1) > priority_weight(tier)

    def test_tier5_lowest(self):
        assert priority_weight(5) == 0.6
        for tier in [1, 2, 3, 4]:
            assert priority_weight(5) < priority_weight(tier)

    def test_unknown_tier_defaults_to_1(self):
        assert priority_weight(99) == 1.0
        assert priority_weight(0) == 1.0


# ===========================================================================
# 5. TestCabinMatchScore
# ===========================================================================

class TestCabinMatchScore:
    def test_all_matched(self):
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business"},
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        score = RewardComputer._cabin_match_score(bookings, PASSENGERS)
        assert score == pytest.approx(1.0)

    def test_none_matched(self):
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "economy"},   # business -> economy
            "P2": {"flight_id": "FL-A", "cabin": "economy"},   # business -> economy
            "P3": {"flight_id": "FL-A", "cabin": "business"},  # economy -> business
            "P4": {"flight_id": "FL-A", "cabin": "business"},  # economy -> business
            "P5": {"flight_id": "FL-A", "cabin": "business"},  # economy -> business
        }
        score = RewardComputer._cabin_match_score(bookings, PASSENGERS)
        assert score == pytest.approx(0.0)

    def test_partial_match_priority_weighted(self):
        """Only P1 (tier 1, weight 1.5) matched. Others mismatched."""
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business"},  # match
            "P2": {"flight_id": "FL-A", "cabin": "economy"},   # mismatch
            "P3": {"flight_id": "FL-A", "cabin": "business"},  # mismatch
            "P4": {"flight_id": "FL-A", "cabin": "business"},  # mismatch
            "P5": {"flight_id": "FL-A", "cabin": "business"},  # mismatch
        }
        total_weight = sum(priority_weight(p["priority_tier"]) for p in PASSENGERS.values())
        expected = priority_weight(1) / total_weight
        score = RewardComputer._cabin_match_score(bookings, PASSENGERS)
        assert score == pytest.approx(expected)


# ===========================================================================
# 6. TestGroupIntegrityScore
# ===========================================================================

class TestGroupIntegrityScore:
    def test_all_same_flight_same_cabin(self):
        bookings = {
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "business"},
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        score, violations = RewardComputer._group_integrity_score(bookings, PASSENGERS, GROUPS)
        # G1: same flight same cabin (both business) -> 1.0
        # But wait, P3 original_cabin is economy, booked as business — doesn't matter for group integrity
        # G2: same flight same cabin -> 1.0
        assert score == pytest.approx(GROUP_SAME_FLIGHT_SAME_CABIN)
        assert violations == 0

    def test_same_flight_diff_cabin(self):
        bookings = {
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},   # diff cabin
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        score, violations = RewardComputer._group_integrity_score(bookings, PASSENGERS, GROUPS)
        # G1: same flight diff cabin -> 0.7
        # G2: same flight same cabin -> 1.0
        expected = (GROUP_SAME_FLIGHT_DIFF_CABIN + GROUP_SAME_FLIGHT_SAME_CABIN) / 2
        assert score == pytest.approx(expected)
        assert violations == 0

    def test_hard_group_split_flights(self):
        bookings = {
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-B", "cabin": "economy"},   # G1 split across flights!
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        score, violations = RewardComputer._group_integrity_score(bookings, PASSENGERS, GROUPS)
        # G1 (hard): split flights -> 0.0 + 1 violation
        # G2 (soft): same flight same cabin -> 1.0
        expected = (GROUP_SPLIT_FLIGHTS_HARD + GROUP_SAME_FLIGHT_SAME_CABIN) / 2
        assert score == pytest.approx(expected)
        assert violations == 1

    def test_soft_group_split_flights(self):
        bookings = {
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},
            "P4": {"flight_id": "FL-A", "cabin": "economy"},
            "P5": {"flight_id": "FL-B", "cabin": "economy"},   # G2 split across flights
        }
        score, violations = RewardComputer._group_integrity_score(bookings, PASSENGERS, GROUPS)
        # G1 (hard): same flight diff cabin -> 0.7
        # G2 (soft): split flights -> 0.4 (no hard violation)
        expected = (GROUP_SAME_FLIGHT_DIFF_CABIN + GROUP_SPLIT_FLIGHTS_SOFT) / 2
        assert score == pytest.approx(expected)
        assert violations == 0  # soft group split is not a hard violation

    def test_no_groups(self):
        bookings = {"P1": {"flight_id": "FL-A", "cabin": "business"}}
        pax = {"P1": PASSENGERS["P1"]}
        score, violations = RewardComputer._group_integrity_score(bookings, pax, {})
        assert score == pytest.approx(1.0)
        assert violations == 0


# ===========================================================================
# 7. TestDeadlineScore
# ===========================================================================

class TestDeadlineScore:
    def test_all_deadlines_met(self):
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business"},  # dl 14:00, arr 12:00 OK
            "P5": {"flight_id": "FL-A", "cabin": "economy"},   # dl 16:00, arr 12:00 OK
        }
        pax = {"P1": PASSENGERS["P1"], "P5": PASSENGERS["P5"]}
        score = RewardComputer._deadline_score(bookings, pax, FLIGHTS)
        assert score == pytest.approx(1.0)

    def test_deadline_missed(self):
        bookings = {
            "P1": {"flight_id": "FL-B", "cabin": "business"},  # dl 14:00, arr 16:00 MISS
            "P5": {"flight_id": "FL-A", "cabin": "economy"},   # dl 16:00, arr 12:00 OK
        }
        pax = {"P1": PASSENGERS["P1"], "P5": PASSENGERS["P5"]}
        score = RewardComputer._deadline_score(bookings, pax, FLIGHTS)
        # P1 missed, P5 met. Priority-weighted.
        total_w = priority_weight(1) + priority_weight(5)
        met_w = priority_weight(5)
        assert score == pytest.approx(met_w / total_w)

    def test_no_deadline_passengers(self):
        pax = {"P2": PASSENGERS["P2"], "P3": PASSENGERS["P3"]}  # no deadlines
        bookings = {
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},
        }
        score = RewardComputer._deadline_score(bookings, pax, FLIGHTS)
        assert score == pytest.approx(1.0)


# ===========================================================================
# 8. TestSSRIntegrity
# ===========================================================================

class TestSSRIntegrity:
    def test_no_violations(self):
        bookings = {
            "P1": {"flight_id": "FL-A", "cabin": "business"},  # UM -> FL-A supports UM
            "P5": {"flight_id": "FL-A", "cabin": "economy"},   # WCHR -> FL-A supports WCHR
        }
        pax = {"P1": PASSENGERS["P1"], "P5": PASSENGERS["P5"]}
        score, violations = RewardComputer._ssr_integrity_score(bookings, pax, FLIGHTS)
        assert score == pytest.approx(1.0)
        assert violations == 0

    def test_one_violation(self):
        bookings = {
            "P1": {"flight_id": "FL-B", "cabin": "business"},  # UM -> FL-B doesn't support UM!
            "P5": {"flight_id": "FL-A", "cabin": "economy"},
        }
        pax = {"P1": PASSENGERS["P1"], "P5": PASSENGERS["P5"]}
        score, violations = RewardComputer._ssr_integrity_score(bookings, pax, FLIGHTS)
        assert violations == 1
        assert score == pytest.approx(0.75)  # 1.0 - 0.25*1

    def test_no_ssr_passengers(self):
        pax = {"P2": PASSENGERS["P2"], "P3": PASSENGERS["P3"]}
        bookings = {
            "P2": {"flight_id": "FL-A", "cabin": "business"},
            "P3": {"flight_id": "FL-A", "cabin": "economy"},
        }
        score, violations = RewardComputer._ssr_integrity_score(bookings, pax, FLIGHTS)
        assert score == pytest.approx(1.0)
        assert violations == 0


# ===========================================================================
# 9. TestGraderWeightsSum
# ===========================================================================

def test_grader_weights_sum_to_one():
    total = (GRADER_W_COVERAGE + GRADER_W_CABIN_MATCH +
             GRADER_W_GROUP_INTEGRITY + GRADER_W_DEADLINE + GRADER_W_SSR_INTEGRITY)
    assert total == pytest.approx(1.0)
