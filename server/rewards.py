"""
Reward computation for the Flight Rebooking Environment.

All reward logic lives in RewardComputer. The class is stateless — it
receives data as arguments and returns (reward_value, reason_string) tuples.
Constants are defined at module level so they can be imported by tests.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-4

PRIORITY_WEIGHTS = {1: 1.5, 2: 1.3, 3: 1.0, 4: 0.8, 5: 0.6}

# Step rewards — info tools
REWARD_LIST_PASSENGERS_FIRST = 0.02
REWARD_LIST_PASSENGERS_CHURN = -0.01  # 3rd+ call with no intervening bookings
REWARD_GET_DETAILS_UNBOOKED = 0.02
REWARD_GET_DETAILS_BOOKED = -0.01
REWARD_LIST_FLIGHTS = 0.01
REWARD_GET_FLIGHT_DETAILS = 0.01

# Step rewards — booking outcomes
REWARD_SAME_CABIN_GROUP = 0.3       # same cabin, whole group together
REWARD_UPGRADE = 0.10                # upgrade from original cabin
REWARD_SPLIT_CABIN_SAME_FLIGHT = -0.02  # split across cabins, same flight (group fallback)
REWARD_DOWNGRADE = -0.02              # cabin downgrade
REWARD_DEADLINE_BONUS = 0.05         # additional bonus for meeting deadline
REWARD_HARD_VIOLATION = -0.30        # SSR mismatch, hard group split, deadline miss with alt
REWARD_FAILED_BOOKING = -0.50        # rejected by environment

# Step rewards — other
REWARD_INVALID_TOOL = -0.20
REWARD_FINALIZE = 0.0                # finalize itself has no step reward

# Grader component weights
GRADER_W_COVERAGE = 0.35
GRADER_W_CABIN_MATCH = 0.15
GRADER_W_GROUP_INTEGRITY = 0.15
GRADER_W_DEADLINE = 0.15
GRADER_W_SSR_INTEGRITY = 0.20

# Hard-constraint penalty (subtracted from final grader score per violation)
GRADER_HARD_PENALTY = 0.15

# Group integrity scores
GROUP_SAME_FLIGHT_SAME_CABIN = 0.7
GROUP_SAME_FLIGHT_DIFF_CABIN = 0.5
GROUP_SPLIT_FLIGHTS_HARD = 0.0
GROUP_SPLIT_FLIGHTS_SOFT = 0.04


def priority_weight(tier: int) -> float:
    """Return the priority multiplier for a given tier."""
    return PRIORITY_WEIGHTS.get(tier, 1.0)


# Cabin ordering for upgrade/downgrade detection
_CABIN_RANK = {"economy": 0, "premium_economy": 1, "business": 2}


def _cabin_rank(cabin: str) -> int:
    return _CABIN_RANK.get(cabin, 0)


# ---------------------------------------------------------------------------
# RewardComputer
# ---------------------------------------------------------------------------

class RewardComputer:
    """
    Stateless reward computer. Instantiated once per episode with the
    episode-level parameters needed for reward scaling.
    """

    def __init__(self, total_passengers: int, max_steps: int):
        self.total_passengers = total_passengers
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Step-level: info calls
    # ------------------------------------------------------------------

    def reward_for_info_call(
        self, tool_name: str, ep_state
    ) -> Tuple[float, str]:
        """Compute reward for an information-gathering tool call."""

        if tool_name == "list_passengers":
            count = ep_state.info_calls.get("list_passengers", 0)
            # Churn: 3rd+ call with no bookings since last list_passengers
            if count >= 3 and ep_state.last_booking_step < ep_state.step_count - (count - 1):
                return (REWARD_LIST_PASSENGERS_CHURN,
                        "Redundant list_passengers call with no intervening bookings")
            if count == 1:
                return (REWARD_LIST_PASSENGERS_FIRST,
                        "First list_passengers call — good planning")
            return (REWARD_LIST_PASSENGERS_FIRST,
                    "list_passengers call")

        if tool_name == "get_passenger_details":
            return (REWARD_GET_DETAILS_UNBOOKED,
                    "Fetched passenger details")

        if tool_name == "get_passenger_details_booked":
            return (REWARD_GET_DETAILS_BOOKED,
                    "Fetched details for already-booked passenger")

        if tool_name == "list_alternative_flights":
            return (REWARD_LIST_FLIGHTS,
                    "Listed alternative flights")

        if tool_name == "get_flight_details":
            return (REWARD_GET_FLIGHT_DETAILS,
                    "Fetched flight details")

        return (0.0, f"Info call: {tool_name}")

    # ------------------------------------------------------------------
    # Step-level: booking
    # ------------------------------------------------------------------

    def reward_for_booking(
        self,
        tool_result: dict,
        passenger: dict,
        ep_state,
    ) -> Tuple[float, str]:
        """Compute reward for a book_passenger call that succeeded."""
        if tool_result["status"] != "success":
            return (REWARD_FAILED_BOOKING,
                    f"Booking failed: {tool_result.get('message', '')}")

        pw = priority_weight(passenger["priority_tier"])
        assigned_cabin = tool_result["cabin"]
        original_cabin = passenger["original_cabin"]

        # Cabin comparison
        assigned_rank = _cabin_rank(assigned_cabin)
        original_rank = _cabin_rank(original_cabin)

        if assigned_cabin == original_cabin:
            reward = REWARD_SAME_CABIN_GROUP * pw
            reason = "Booking: same cabin"
        elif assigned_rank > original_rank:
            reward = REWARD_UPGRADE * pw
            reason = "Booking: cabin upgrade"
        else:
            reward = REWARD_DOWNGRADE * pw
            reason = "Booking: cabin downgrade"

        # Deadline bonus
        if tool_result.get("deadline_met"):
            reward += REWARD_DEADLINE_BONUS * pw
            reason += " + deadline met"

        return (reward, reason)

    def reward_for_group_booking(
        self,
        tool_result: dict,
        group_passengers: List[dict],
        ep_state,
    ) -> Tuple[float, str]:
        """Compute reward for a book_group call that succeeded."""
        if tool_result["status"] != "success":
            return (REWARD_FAILED_BOOKING,
                    f"Group booking failed: {tool_result.get('message', '')}")

        total_reward = 0.0
        booked_list = tool_result["booked"]
        cabin_set = set()

        for entry in booked_list:
            pid = entry["passenger_id"]
            pax = next(p for p in group_passengers if p["passenger_id"] == pid)
            pw = priority_weight(pax["priority_tier"])

            assigned_cabin = entry["cabin"]
            original_cabin = pax["original_cabin"]
            cabin_set.add(assigned_cabin)

            assigned_rank = _cabin_rank(assigned_cabin)
            original_rank = _cabin_rank(original_cabin)

            if assigned_cabin == original_cabin:
                total_reward += REWARD_SAME_CABIN_GROUP * pw
            elif assigned_rank > original_rank:
                total_reward += REWARD_UPGRADE * pw
            else:
                total_reward += REWARD_DOWNGRADE * pw

            # Deadline bonus
            if pax.get("downstream_deadline"):
                total_reward += REWARD_DEADLINE_BONUS * pw

        if len(cabin_set) == 1:
            reason = "Group booking: all same cabin on same flight"
        else:
            reason = "Group booking: split cabin on same flight"

        return (total_reward, reason)

    def reward_for_failed_action(
        self, tool_result: dict
    ) -> Tuple[float, str]:
        """Reward for a booking action rejected by the environment."""
        return (REWARD_FAILED_BOOKING,
                f"Action failed: {tool_result.get('message', '')}")

    def reward_for_invalid_tool(self) -> Tuple[float, str]:
        """Reward when the agent submits an unrecognized tool name."""
        return (REWARD_INVALID_TOOL, "Unrecognized tool name")

    # ------------------------------------------------------------------
    # Terminal: grader score
    # ------------------------------------------------------------------

    def grader_score(
        self,
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        flights: Dict[str, dict],
        groups: Dict[str, List[str]],
    ) -> float:
        """
        Compute the 0-1 grader score for hackathon evaluation.

        Components:
          coverage_score      (0.35) — fraction of passengers booked
          cabin_match_score   (0.15) — priority-weighted cabin match fraction
          group_integrity     (0.15) — per-group score, averaged
          deadline_score      (0.15) — priority-weighted deadline-met fraction
          ssr_integrity       (0.20) — 1.0 minus per-violation penalty
        """
        breakdown = self.terminal_breakdown(bookings, passengers, flights, groups)

        score = (
            GRADER_W_COVERAGE * breakdown["coverage_score"]
            + GRADER_W_CABIN_MATCH * breakdown["cabin_match_score"]
            + GRADER_W_GROUP_INTEGRITY * breakdown["group_integrity_score"]
            + GRADER_W_DEADLINE * breakdown["deadline_score"]
            + GRADER_W_SSR_INTEGRITY * breakdown["ssr_integrity_score"]
        )

        # Hard-constraint penalties
        score -= GRADER_HARD_PENALTY * breakdown["hard_violations"]

        return max(EPS, min(1.0 - EPS, score))

    def terminal_breakdown(
        self,
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        flights: Dict[str, dict],
        groups: Dict[str, List[str]],
    ) -> dict:
        """
        Compute all grader sub-scores and return them in a dict.
        """
        n_total = len(passengers)

        # --- Coverage ---
        n_booked = len(bookings)
        coverage_score = n_booked / n_total if n_total > 0 else 0.0

        # --- Cabin match (priority-weighted) ---
        cabin_match_score = self._cabin_match_score(bookings, passengers)

        # --- Group integrity ---
        group_integrity_score, hard_violations = self._group_integrity_score(
            bookings, passengers, groups
        )

        # --- Deadline score (priority-weighted) ---
        deadline_score = self._deadline_score(bookings, passengers, flights)

        # --- SSR integrity ---
        ssr_integrity_score, ssr_violations = self._ssr_integrity_score(
            bookings, passengers, flights
        )
        hard_violations += ssr_violations

        return {
            "coverage_score": coverage_score,
            "cabin_match_score": cabin_match_score,
            "group_integrity_score": group_integrity_score,
            "deadline_score": deadline_score,
            "ssr_integrity_score": ssr_integrity_score,
            "hard_violations": hard_violations,
        }

    # ------------------------------------------------------------------
    # Grader sub-score helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cabin_match_score(
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
    ) -> float:
        """Priority-weighted fraction of booked passengers in their original cabin."""
        total_weight = 0.0
        matched_weight = 0.0

        for pid, pax in passengers.items():
            pw = priority_weight(pax["priority_tier"])
            total_weight += pw
            if pid in bookings and bookings[pid]["cabin"] == pax["original_cabin"]:
                matched_weight += pw

        return matched_weight / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _group_integrity_score(
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        groups: Dict[str, List[str]],
    ) -> Tuple[float, int]:
        """
        Average per-group integrity score.

        Returns (avg_score, hard_violation_count).
        If no groups exist, returns (1.0, 0).
        """
        if not groups:
            return 1.0, 0

        total_score = 0.0
        hard_violations = 0

        for gid, member_ids in groups.items():
            integrity = passengers[member_ids[0]]["group_integrity"]

            # Collect flights and cabins for booked members
            booked_flights = set()
            booked_cabins = set()
            all_booked = True
            for pid in member_ids:
                if pid in bookings:
                    booked_flights.add(bookings[pid]["flight_id"])
                    booked_cabins.add(bookings[pid]["cabin"])
                else:
                    all_booked = False

            if not booked_flights:
                # No members booked — 0 score, no hard violation
                total_score += 0.0
                continue

            if not all_booked:
                # Partially booked — treat as split
                if integrity == "hard":
                    total_score += GROUP_SPLIT_FLIGHTS_HARD
                    hard_violations += 1
                else:
                    total_score += GROUP_SPLIT_FLIGHTS_SOFT
                continue

            if len(booked_flights) == 1 and len(booked_cabins) == 1:
                total_score += GROUP_SAME_FLIGHT_SAME_CABIN
            elif len(booked_flights) == 1:
                total_score += GROUP_SAME_FLIGHT_DIFF_CABIN
            else:
                # Split across flights
                if integrity == "hard":
                    total_score += GROUP_SPLIT_FLIGHTS_HARD
                    hard_violations += 1
                else:
                    total_score += GROUP_SPLIT_FLIGHTS_SOFT

        avg = total_score / len(groups) if groups else 1.0
        return avg, hard_violations

    @staticmethod
    def _deadline_score(
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        flights: Dict[str, dict],
    ) -> float:
        """Priority-weighted fraction of deadline-bearing passengers whose deadlines are met."""
        total_weight = 0.0
        met_weight = 0.0

        for pid, pax in passengers.items():
            if pax["downstream_deadline"] is None:
                continue
            pw = priority_weight(pax["priority_tier"])
            total_weight += pw

            if pid in bookings:
                fl = flights[bookings[pid]["flight_id"]]
                from server.tools import meets_deadline
                if meets_deadline(fl["arrival_time"], pax["downstream_deadline"]):
                    met_weight += pw

        return met_weight / total_weight if total_weight > 0 else 1.0

    @staticmethod
    def _ssr_integrity_score(
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        flights: Dict[str, dict],
    ) -> Tuple[float, int]:
        """
        1.0 if no SSR violations; penalised per violation.

        Returns (score, violation_count).
        """
        violations = 0

        for pid, pax in passengers.items():
            if not pax["ssr_flags"]:
                continue
            if pid not in bookings:
                continue

            fl = flights[bookings[pid]["flight_id"]]
            required = set(pax["ssr_flags"])
            supported = set(fl["supports_ssr"])
            if not required.issubset(supported):
                violations += 1

        # Each violation subtracts 0.25 from 1.0
        score = max(0.0, 1.0 - 0.25 * violations)
        return score, violations
