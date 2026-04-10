"""
Reward computation for the Airline Seat Reassignment Environment.

All reward logic lives in RewardComputer. The class is stateless — it
receives data as arguments and returns (reward_value, reason_string) tuples.
Constants are defined at module level so they can be imported by tests.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------
REWARD_FETCH_NEUTRAL      =  0.0
REWARD_FETCH_REDUNDANT    = -0.05
REWARD_FETCH_ERROR        = -0.1

REWARD_ASSIGN_CABIN_PREF    =  0.35   # cabin match + window preference satisfied
REWARD_ASSIGN_CABIN_NOPREF  =  0.2    # cabin match + no window preference
REWARD_ASSIGN_CABIN_PREFMISS =  0.1   # cabin match + window preference missed
REWARD_ASSIGN_CABIN_MISMATCH = -0.1   # wrong cabin
REWARD_ASSIGN_ERROR          = -0.4

REWARD_SWAP_IMPROVE  =  0.25
REWARD_SWAP_NEUTRAL  = -0.05
REWARD_SWAP_WORSE    = -0.15
REWARD_SWAP_ERROR    = -0.4

REWARD_INVALID_TOOL       = -0.4
REWARD_INCOMPLETE_PENALTY = -1.0

TERMINAL_W_CABIN =  1.5
TERMINAL_W_PREF  =  1.0
TERMINAL_W_EFF   =  0.5


# ---------------------------------------------------------------------------
# RewardComputer
# ---------------------------------------------------------------------------

class RewardComputer:
    """
    Stateless reward computer. Instantiated once per episode with the
    episode-level parameters needed for efficiency scoring.
    """

    def __init__(self, total_passengers: int, max_steps: int):
        self.total_passengers = total_passengers
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Per-step rewards
    # ------------------------------------------------------------------

    def reward_for_fetch(self, is_redundant: bool, is_error: bool) -> tuple[float, str]:
        """Reward for a get_passenger_details call."""
        if is_error:
            return (REWARD_FETCH_ERROR, "Fetch error: invalid seat or passenger already reassigned")
        if is_redundant:
            return (REWARD_FETCH_REDUNDANT, "Redundant fetch: seat already queried this episode")
        return (REWARD_FETCH_NEUTRAL, "Fetch: new seat queried")

    def reward_for_assign(self, tool_result: dict) -> tuple[float, str]:
        """
        Reward for an assign_seat call.

        Checks cabin_match first; preference satisfaction is only considered
        when the cabin is correct. Handles both single (window) and dual
        (window + legroom) preference dimensions gracefully.
        """
        if tool_result["status"] == "error":
            return (REWARD_ASSIGN_ERROR,
                    f"Assignment error: {tool_result.get('message', '')}")

        cabin_match = tool_result["cabin_match"]
        if not cabin_match:
            return (REWARD_ASSIGN_CABIN_MISMATCH, "Assignment: cabin mismatch")

        # Collect active preference results (True/False only; None = not applicable)
        window_pref  = tool_result.get("window_preference_satisfied")
        legroom_pref = tool_result.get("legroom_preference_satisfied")
        active_prefs = [p for p in (window_pref, legroom_pref) if p is not None]

        if not active_prefs:
            # No paid preferences at all (easy task, or passengers without any pref)
            return (REWARD_ASSIGN_CABIN_NOPREF,
                    "Valid assignment: cabin match, no preferences")

        if all(active_prefs):
            return (REWARD_ASSIGN_CABIN_PREF,
                    "Valid assignment: cabin match, all preferences satisfied")

        if not any(active_prefs):
            return (REWARD_ASSIGN_CABIN_PREFMISS,
                    "Valid assignment: cabin match, preferences not satisfied")

        # Mixed: some satisfied, some missed
        return (REWARD_ASSIGN_CABIN_NOPREF,
                "Valid assignment: cabin match, some preferences satisfied")

    def reward_for_swap(
        self,
        tool_result: dict,
        pax1_info: dict,
        pax2_info: dict,
        old_seat_1_info: dict,
        old_seat_2_info: dict,
        new_seat_1_info: dict,
        new_seat_2_info: dict,
    ) -> tuple[float, str]:
        """
        Reward for a swap_seats call.

        Computes the combined constraint score for both passengers before
        and after the swap; rewards are based on the sign of the delta.
        """
        if tool_result["status"] == "error":
            return (REWARD_SWAP_ERROR,
                    f"Swap error: {tool_result.get('message', '')}")

        score_before = (
            self._constraint_score(pax1_info, old_seat_1_info)
            + self._constraint_score(pax2_info, old_seat_2_info)
        )
        score_after = (
            self._constraint_score(pax1_info, new_seat_1_info)
            + self._constraint_score(pax2_info, new_seat_2_info)
        )
        delta = score_after - score_before

        if delta > 0:
            return (REWARD_SWAP_IMPROVE, "Swap improved constraint satisfaction")
        if delta < 0:
            return (REWARD_SWAP_WORSE, "Swap worsened constraint satisfaction")
        return (REWARD_SWAP_NEUTRAL, "Swap did not change constraint satisfaction")

    def reward_for_invalid_tool(self) -> tuple[float, str]:
        """Reward when the agent submits an unrecognized tool name."""
        return (REWARD_INVALID_TOOL, "Unrecognized tool name")

    # ------------------------------------------------------------------
    # Terminal reward (end of episode)
    # ------------------------------------------------------------------

    def terminal_reward(
        self,
        assignments_df: pd.DataFrame,
        passengers_df: pd.DataFrame,
        ac2_seat_info: dict,
        total_steps: int,
    ) -> tuple[float, dict]:
        """
        Compute the terminal reward at episode end.

        Returns (reward_value, breakdown_dict).  The breakdown contains
        individual component scores for logging/observation.
        """
        merged = self._merged(assignments_df, passengers_df)
        n_total = len(merged)

        assigned = merged[merged["seat_ac2"].notna()].copy()
        n_assigned = len(assigned)

        cabin_score = self._cabin_score(assigned, ac2_seat_info, n_total)
        preference_score = self._preference_score(merged, assigned, ac2_seat_info)
        efficiency_score = max(0.0, 1.0 - total_steps / self.max_steps)
        incomplete_penalty = REWARD_INCOMPLETE_PENALTY if n_assigned < n_total else 0.0

        weighted_total = (
            TERMINAL_W_CABIN * cabin_score
            + TERMINAL_W_PREF * preference_score
            + TERMINAL_W_EFF * efficiency_score
            + incomplete_penalty
        )

        breakdown = {
            "cabin_score": cabin_score,
            "preference_score": preference_score,
            "efficiency_score": efficiency_score,
            "weighted_total": weighted_total,
            "incomplete_penalty": incomplete_penalty,
        }
        return weighted_total, breakdown

    # ------------------------------------------------------------------
    # Grader score (evaluation metric, not used during episode)
    # ------------------------------------------------------------------

    def grader_score(
        self,
        assignments_df: pd.DataFrame,
        passengers_df: pd.DataFrame,
        ac2_seat_info: dict,
    ) -> float:
        """
        Return a 0.0–1.0 quality score for use by hackathon evaluation.

        Unassigned passengers count as zero contribution to both cabin and
        preference scores.  If there are no paid-window passengers,
        the score equals the cabin score alone.
        """
        merged = self._merged(assignments_df, passengers_df)
        n_total = len(merged)
        if n_total == 0:
            score = 0.0
        else:
            assigned = merged[merged["seat_ac2"].notna()].copy()

            cabin_score = self._cabin_score(assigned, ac2_seat_info, n_total)

            # Check whether any preference columns exist with at least one paid passenger
            has_window_pref  = (
                "paid_window" in merged.columns
                and bool(merged["paid_window"].astype(bool).sum() > 0)
            )
            has_legroom_pref = (
                "paid_legroom" in merged.columns
                and bool(merged["paid_legroom"].astype(bool).sum() > 0)
            )

            if not has_window_pref and not has_legroom_pref:
                score = cabin_score
            else:
                preference_score = self._preference_score(merged, assigned, ac2_seat_info)
                score = (cabin_score + preference_score) / 2.0

        EPS = 1e-4
        return min(max(score, EPS), 1.0 - EPS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _constraint_score(self, passenger: dict, seat_info: dict) -> float:
        """
        Single float measuring how well a passenger-seat pairing satisfies
        constraints.

        +1.0 for correct cabin.
        +1.0 if paid_window and seat is window; -0.5 if paid_window and not window.
        +1.0 if paid_legroom and seat has extra_legroom; -0.5 if not.
        No bonus/penalty for dimensions the passenger did not pay for.
        """
        score = 0.0
        if passenger.get("cabin") == seat_info.get("cabin"):
            score += 1.0
        if passenger.get("paid_window", False):
            score += 1.0 if seat_info.get("seat_type") == "window" else -0.5
        if passenger.get("paid_legroom", False):
            score += 1.0 if seat_info.get("extra_legroom", False) else -0.5
        return score

    # ------------------------------------------------------------------
    # Private computation helpers shared by terminal_reward & grader_score
    # ------------------------------------------------------------------

    @staticmethod
    def _merged(assignments_df: pd.DataFrame, passengers_df: pd.DataFrame) -> pd.DataFrame:
        """Join assignments onto passengers, both normalised to passenger_id index."""
        asgn = (
            assignments_df
            if assignments_df.index.name == "passenger_id"
            else assignments_df.set_index("passenger_id")
        )
        pax = (
            passengers_df.set_index("passenger_id")
            if passengers_df.index.name != "passenger_id"
            else passengers_df
        )
        return pax.join(asgn[["seat_ac2"]], how="left")

    @staticmethod
    def _cabin_score(assigned: pd.DataFrame, ac2_seat_info: dict, n_total: int) -> float:
        """Fraction of all passengers assigned to a matching-cabin seat."""
        if assigned.empty or n_total == 0:
            return 0.0
        cabin_map = pd.Series({s: info["cabin"] for s, info in ac2_seat_info.items()})
        assigned = assigned.copy()
        assigned["ac2_cabin"] = assigned["seat_ac2"].map(cabin_map)
        return int((assigned["cabin"] == assigned["ac2_cabin"]).sum()) / n_total

    @staticmethod
    def _preference_score(
        merged: pd.DataFrame,
        assigned: pd.DataFrame,
        ac2_seat_info: dict,
    ) -> float:
        """
        Average satisfaction rate across all active preference dimensions.

        For each dimension (window, legroom):
          - Compute fraction of paid passengers who got the preference satisfied.
          - Only include dimensions where at least one passenger paid for it.
        Returns 1.0 if no preference dimension is active (easy task or no paid pref).
        Unassigned paid passengers count as unsatisfied.
        """
        scores = []

        # --- Window preference ---
        if "paid_window" in merged.columns:
            n_window = int(merged["paid_window"].astype(bool).sum())
            if n_window > 0:
                type_map = pd.Series(
                    {s: info.get("seat_type", "") for s, info in ac2_seat_info.items()}
                )
                win_assigned = assigned[assigned["paid_window"].astype(bool)].copy()
                if win_assigned.empty:
                    scores.append(0.0)
                else:
                    win_assigned["ac2_seat_type"] = win_assigned["seat_ac2"].map(type_map)
                    scores.append(
                        int((win_assigned["ac2_seat_type"] == "window").sum()) / n_window
                    )

        # --- Legroom preference ---
        if "paid_legroom" in merged.columns:
            n_legroom = int(merged["paid_legroom"].astype(bool).sum())
            if n_legroom > 0:
                legroom_map = pd.Series(
                    {s: bool(info.get("extra_legroom", False)) for s, info in ac2_seat_info.items()}
                )
                leg_assigned = assigned[assigned["paid_legroom"].astype(bool)].copy()
                if leg_assigned.empty:
                    scores.append(0.0)
                else:
                    leg_assigned["ac2_extra_legroom"] = leg_assigned["seat_ac2"].map(legroom_map)
                    scores.append(
                        int(leg_assigned["ac2_extra_legroom"].astype(bool).sum()) / n_legroom
                    )

        if not scores:
            return 1.0
        return sum(scores) / len(scores)
