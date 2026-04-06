"""
Core environment implementation for the Airline Seat Reassignment task.

The agent interacts via three tools:
  - get_passenger_details  → inspect an AC-1 seat
  - assign_seat            → move a passenger from AC-1 to a specific AC-2 seat
  - swap_seats             → swap two already-assigned AC-2 passengers

An episode ends when all 20 passengers are assigned or max_steps is reached.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

import pandas as pd
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AirlineReassignmentAction, AirlineReassignmentObservation, AirlineReassignmentState
    from .tools import tool_assign_seat, tool_get_passenger_details, tool_swap_seats
    from .rewards import RewardComputer
except ImportError:
    from models import AirlineReassignmentAction, AirlineReassignmentObservation, AirlineReassignmentState
    from server.tools import tool_assign_seat, tool_get_passenger_details, tool_swap_seats
    from server.rewards import RewardComputer


# ---------------------------------------------------------------------------
# Internal episode state (not exposed to the agent)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    # Data loaded from files — immutable during an episode
    ac1_config: dict
    ac2_config: dict
    passengers_df: pd.DataFrame
    seats_ac1_df: pd.DataFrame
    seats_ac2_df: pd.DataFrame

    # Mutable assignment state — the only thing that changes per step
    assignments: pd.DataFrame

    # O(1) lookup structures built at reset time
    passengers_by_id: Dict[str, dict]
    ac1_seat_set: set
    ac2_seat_set: set
    ac1_seat_info: Dict[str, dict]
    ac2_seat_info: Dict[str, dict]

    # Episode tracking
    fetched_seats: set
    step_count: int
    max_steps: int
    cumulative_reward: float
    done: bool


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class AirlineReassignmentEnvironment(Environment):
    """
    OpenEnv environment for airline seat reassignment.

    An episode consists of an agent reassigning 20 passengers from AC-1 to AC-2
    using three tools.  The episode ends when all passengers are reassigned or
    the step limit is reached.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: Optional[EpisodeState] = None
        self._reward_computer: Optional[RewardComputer] = None
        # Data directory is two levels up from this file (project root / data)
        self._data_dir = Path(__file__).resolve().parent.parent / "data"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> AirlineReassignmentObservation:
        """Load data, build lookup dicts, and return the initial observation."""
        episode_id = str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)

        # --- load files ---
        with open(self._data_dir / "ac1_config.json") as f:
            ac1_config = json.load(f)
        with open(self._data_dir / "ac2_config.json") as f:
            ac2_config = json.load(f)

        seats_ac1_df  = pd.read_csv(self._data_dir / "seats_ac1.csv")
        seats_ac2_df  = pd.read_csv(self._data_dir / "seats_ac2.csv")
        passengers_df = pd.read_csv(self._data_dir / "passengers.csv")

        # assignments.csv: seat_ac2 column starts all-NaN (passengers not yet moved)
        assignments_df = (
            pd.read_csv(self._data_dir / "assignments.csv")
            .copy()
            .set_index("passenger_id")
        )

        # --- build O(1) lookup structures ---
        passengers_by_id: Dict[str, dict] = (
            passengers_df.set_index("passenger_id").to_dict("index")
        )
        ac1_seat_set = set(seats_ac1_df["seat_id"])
        ac2_seat_set = set(seats_ac2_df["seat_id"])
        ac1_seat_info: Dict[str, dict] = (
            seats_ac1_df.set_index("seat_id")[["cabin", "seat_type"]].to_dict("index")
        )
        ac2_seat_info: Dict[str, dict] = (
            seats_ac2_df.set_index("seat_id")[["cabin", "seat_type"]].to_dict("index")
        )

        total_passengers = len(passengers_df)
        max_steps = 3 * total_passengers

        self._episode = EpisodeState(
            ac1_config=ac1_config,
            ac2_config=ac2_config,
            passengers_df=passengers_df,
            seats_ac1_df=seats_ac1_df,
            seats_ac2_df=seats_ac2_df,
            assignments=assignments_df,
            passengers_by_id=passengers_by_id,
            ac1_seat_set=ac1_seat_set,
            ac2_seat_set=ac2_seat_set,
            ac1_seat_info=ac1_seat_info,
            ac2_seat_info=ac2_seat_info,
            fetched_seats=set(),
            step_count=0,
            max_steps=max_steps,
            cumulative_reward=0.0,
            done=False,
        )
        self._reward_computer = RewardComputer(
            total_passengers=total_passengers,
            max_steps=max_steps,
        )

        return self._build_observation(
            tool_result=None,
            reward=0.0,
            reward_reason="Episode started",
            done=False,
        )

    def step(self, action: AirlineReassignmentAction) -> AirlineReassignmentObservation:  # type: ignore[override]
        """Execute one tool call and return the updated observation."""
        if self._episode is None:
            raise RuntimeError("step() called without reset()")
        if self._episode.done:
            raise RuntimeError("step() called on a terminated episode")

        ep = self._episode
        ep.step_count += 1
        self._state.step_count += 1

        tool_name = action.tool_name
        args = action.args

        # --- route tool call ---
        try:
            if tool_name == "get_passenger_details":
                seat_id = args.get("seat_id", "")
                # Record redundancy BEFORE the tool adds to fetched_seats
                was_already_fetched = seat_id in ep.fetched_seats
                tool_result = tool_get_passenger_details(ep, seat_id)
                reward, reason = self._reward_computer.reward_for_fetch(
                    is_redundant=was_already_fetched,
                    is_error=tool_result["status"] == "error",
                )

            elif tool_name == "assign_seat":
                tool_result = tool_assign_seat(
                    ep,
                    args.get("passenger_id", ""),
                    args.get("target_seat_id", ""),
                )
                reward, reason = self._reward_computer.reward_for_assign(tool_result)

            elif tool_name == "swap_seats":
                tool_result = tool_swap_seats(
                    ep,
                    args.get("passenger_id_1", ""),
                    args.get("passenger_id_2", ""),
                )
                if tool_result["status"] == "success":
                    s = tool_result["swap"]
                    pax1 = ep.passengers_by_id[s[0]["passenger_id"]]
                    pax2 = ep.passengers_by_id[s[1]["passenger_id"]]
                    reward, reason = self._reward_computer.reward_for_swap(
                        tool_result,
                        pax1, pax2,
                        ep.ac2_seat_info[s[0]["from_seat"]],
                        ep.ac2_seat_info[s[1]["from_seat"]],
                        ep.ac2_seat_info[s[0]["to_seat"]],
                        ep.ac2_seat_info[s[1]["to_seat"]],
                    )
                else:
                    reward, reason = self._reward_computer.reward_for_swap(
                        tool_result, {}, {}, {}, {}, {}, {}
                    )

            else:
                tool_result = {"status": "error", "message": f"Unknown tool: {tool_name!r}"}
                reward, reason = self._reward_computer.reward_for_invalid_tool()

        except Exception as exc:  # unexpected runtime error — keep episode alive
            tool_result = {"status": "error", "message": f"Internal error: {exc}"}
            reward, reason = self._reward_computer.reward_for_invalid_tool()

        # --- check termination ---
        all_assigned = ep.assignments["seat_ac2"].notna().all()
        step_limit_reached = ep.step_count >= ep.max_steps
        done = bool(all_assigned or step_limit_reached)

        # --- grader score (computed every step so agent sees live quality) ---
        grader = self._reward_computer.grader_score(
            assignments_df=ep.assignments,
            passengers_df=ep.passengers_df,
            ac2_seat_info=ep.ac2_seat_info,
        )

        # --- terminal reward ---
        if done:
            terminal_reward, terminal_breakdown = self._reward_computer.terminal_reward(
                assignments_df=ep.assignments,
                passengers_df=ep.passengers_df,
                ac2_seat_info=ep.ac2_seat_info,
                total_steps=ep.step_count,
            )
            reward += terminal_reward

            if step_limit_reached and not all_assigned:
                reason += " | Episode timed out — not all passengers assigned"
            else:
                reason += f" | Episode complete — terminal reward: {terminal_reward:.2f}"

            tool_result = {
                **(tool_result or {}),
                "terminal_breakdown": terminal_breakdown,
            }
            ep.done = True

        ep.cumulative_reward += reward

        return self._build_observation(
            tool_result=tool_result,
            reward=reward,
            reward_reason=reason,
            done=done,
            grader_score=grader,
        )

    @property
    def state(self) -> AirlineReassignmentState:  # type: ignore[override]
        if self._episode is None:
            return AirlineReassignmentState()

        ep = self._episode
        assigned_count = int(ep.assignments["seat_ac2"].notna().sum())

        return AirlineReassignmentState(
            episode_id=self._state.episode_id,
            step_count=ep.step_count,
            total_passengers=len(ep.passengers_df),
            passengers_assigned=assigned_count,
            passengers_remaining=len(ep.passengers_df) - assigned_count,
            cumulative_reward=ep.cumulative_reward,
            is_complete=ep.done,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        tool_result: Optional[dict],
        reward: float,
        reward_reason: str,
        done: bool,
        grader_score: Optional[float] = None,
    ) -> AirlineReassignmentObservation:
        ep = self._episode

        # AC-1 seats whose passengers have NOT yet been moved
        unassigned_mask = ep.assignments["seat_ac2"].isna()
        ac1_seats_occupied = sorted(
            ep.assignments.loc[unassigned_mask, "seat_ac1"].tolist()
        )

        # AC-2 current occupancy: seat_id → passenger_id
        assigned_mask = ~unassigned_mask
        assigned = ep.assignments.loc[assigned_mask]
        ac2_occupied: Dict[str, str] = dict(zip(assigned["seat_ac2"], assigned.index))

        # AC-2 available seats (sorted for deterministic ordering)
        occupied_set = set(ac2_occupied.keys())
        ac2_available = sorted(s for s in ep.ac2_seat_set if s not in occupied_set)

        return AirlineReassignmentObservation(
            ac1_layout=ep.ac1_config,
            ac2_layout=ep.ac2_config,
            ac1_seats_occupied=ac1_seats_occupied,
            ac2_seats_occupied=ac2_occupied,
            ac2_seats_available=ac2_available,
            passengers_remaining=int(unassigned_mask.sum()),
            passengers_total=len(ep.passengers_df),
            tool_result=tool_result,
            reward_reason=reward_reason,
            step_count=ep.step_count,
            max_steps=ep.max_steps,
            cumulative_reward=ep.cumulative_reward,
            grader_score=grader_score,
            done=done,
            reward=reward,
        )
