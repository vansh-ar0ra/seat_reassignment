"""
Reward computation for the Flight Rebooking Environment (plan-then-commit model).

Simplified reward structure:
  - Small cost per tool call (info or finalize)
  - Plan submission reward = grader preview score + per-call cost
  - Penalties for invalid tools, duplicate submissions, and finalizing without a plan

All grader logic (sub-scores, weights, terminal breakdown) is unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-4

PRIORITY_WEIGHTS = {1: 1.5, 2: 1.3, 3: 1.0, 4: 0.8, 5: 0.6}

# Step rewards — plan-then-commit model
REWARD_PER_CALL_COST = 0.0
REWARD_INVALID_TOOL = -0.20
REWARD_NO_PLAN_FINALIZE = -0.10
REWARD_DUPLICATE_SUBMIT = -0.10
REWARD_REPEATED_INFO_CALL = -0.05

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


# ---------------------------------------------------------------------------
# RewardComputer
# ---------------------------------------------------------------------------

class RewardComputer:
    """
    Stateless reward computer for plan-then-commit model.
    """

    def __init__(self, total_passengers: int, max_steps: int):
        self.total_passengers = total_passengers
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Step-level rewards
    # ------------------------------------------------------------------

    def reward_for_info_call(self, repeated: bool = False) -> Tuple[float, str]:
        """Reward for an info-gathering call (get_full_manifest, get_flight_inventory)."""
        if repeated:
            return (REWARD_REPEATED_INFO_CALL, "Repeated info call — penalty applied")
        return (REWARD_PER_CALL_COST, "Information gathered")

    def reward_for_plan_submission(self, plan_grader_preview: float) -> Tuple[float, str]:
        """Reward for submitting a plan. Reward = preview score + per_call_cost."""
        reward = plan_grader_preview + REWARD_PER_CALL_COST
        return (reward, f"Plan submitted (preview: {plan_grader_preview:.4f})")

    def reward_for_duplicate_submit(self) -> Tuple[float, str]:
        """Penalty for attempting a second submit_plan (rejected)."""
        return (REWARD_DUPLICATE_SUBMIT, "Plan already submitted — no revisions allowed")

    def reward_for_finalize(self, has_plan: bool) -> Tuple[float, str]:
        """Reward for finalizing. Penalty if no plan was submitted."""
        if not has_plan:
            return (REWARD_NO_PLAN_FINALIZE, "Finalized without a plan")
        return (REWARD_PER_CALL_COST, "Plan finalized")

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
