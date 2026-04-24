"""
Core environment implementation for the Flight Rebooking task.

A flight has been cancelled. The agent must rebook passengers onto
alternative flights using 7 tools:
  - list_passengers           -> survey all passengers
  - get_passenger_details     -> inspect one passenger
  - list_alternative_flights  -> survey flight inventory
  - get_flight_details        -> inspect one flight
  - book_passenger            -> commit one passenger to a flight/cabin
  - book_group                -> commit an entire group atomically
  - finalize_plan             -> end the episode and trigger grading

An episode ends when finalize_plan is called, all passengers are booked,
or the step limit is reached.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        FlightRebookingAction,
        FlightRebookingObservation,
        FlightRebookingState,
    )
    from .tools import (
        tool_list_passengers,
        tool_get_passenger_details,
        tool_list_alternative_flights,
        tool_get_flight_details,
        tool_book_passenger,
        tool_book_group,
        tool_finalize_plan,
    )
    from .rewards import RewardComputer
except ImportError:
    from models import (
        FlightRebookingAction,
        FlightRebookingObservation,
        FlightRebookingState,
    )
    from server.tools import (
        tool_list_passengers,
        tool_get_passenger_details,
        tool_list_alternative_flights,
        tool_get_flight_details,
        tool_book_passenger,
        tool_book_group,
        tool_finalize_plan,
    )
    from server.rewards import RewardComputer


# ---------------------------------------------------------------------------
# Internal episode state (not exposed to the agent)
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    # Immutable data loaded from files
    passengers: Dict[str, dict] = field(default_factory=dict)
    flights: Dict[str, dict] = field(default_factory=dict)
    groups: Dict[str, List[str]] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    # Mutable booking state
    bookings: Dict[str, dict] = field(default_factory=dict)
    flight_availability: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Tracking for reward computation
    info_calls: Dict[str, int] = field(default_factory=dict)
    last_booking_step: int = 0
    passenger_details_fetched: Set[str] = field(default_factory=set)

    # Whether list_alternative_flights has been called (for snapshot in obs)
    flights_listed: bool = False

    # Episode metadata
    task_id: str = ""
    step_count: int = 0
    max_steps: int = 0
    cumulative_reward: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FlightRebookingEnvironment(Environment):
    """
    OpenEnv environment for airline flight rebooking after cancellation.

    An episode consists of an agent rebooking passengers from a cancelled
    flight onto alternative flights using 7 tools. The episode ends when
    finalize_plan is called, all passengers are booked, or the step limit
    is reached.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: Optional[EpisodeState] = None
        self._reward_computer: Optional[RewardComputer] = None
        self._data_dir = Path(__file__).resolve().parent.parent / "data"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, task_id: str = "medium"
    ) -> FlightRebookingObservation:
        """Load data, build lookup structures, and return the initial observation."""
        episode_id = str(uuid4())
        self._state = State(episode_id=episode_id, step_count=0)

        # Resolve data directory
        task_dir = self._data_dir / task_id
        if not task_dir.is_dir():
            raise ValueError(
                f"Unknown task_id {task_id!r}: no data directory at {task_dir}"
            )

        # Load files
        with open(task_dir / "config.json") as f:
            config = json.load(f)
        with open(task_dir / "passengers.json") as f:
            passengers_list = json.load(f)["passengers"]
        with open(task_dir / "flights.json") as f:
            flights_list = json.load(f)["flights"]

        # Build dicts
        passengers = {p["passenger_id"]: p for p in passengers_list}
        flights = {fl["flight_id"]: fl for fl in flights_list}

        # Build groups
        groups: Dict[str, List[str]] = {}
        for pid, pax in passengers.items():
            gid = pax.get("group_id")
            if gid:
                groups.setdefault(gid, []).append(pid)

        # Deep copy availability so bookings can decrement without mutating source
        flight_availability = {
            fid: copy.deepcopy(fl["cabin_availability"])
            for fid, fl in flights.items()
        }

        max_steps = config.get("max_steps", 60)

        self._episode = EpisodeState(
            passengers=passengers,
            flights=flights,
            groups=groups,
            config=config,
            bookings={},
            flight_availability=flight_availability,
            info_calls={},
            last_booking_step=0,
            passenger_details_fetched=set(),
            flights_listed=False,
            task_id=task_id,
            step_count=0,
            max_steps=max_steps,
            cumulative_reward=0.0,
            done=False,
        )
        self._reward_computer = RewardComputer(
            total_passengers=len(passengers),
            max_steps=max_steps,
        )

        return self._build_observation(
            tool_result=None,
            reward=0.0,
            reward_reason="Episode started",
            done=False,
        )

    def step(
        self, action: FlightRebookingAction  # type: ignore[override]
    ) -> FlightRebookingObservation:
        """Execute one tool call and return the updated observation."""
        if self._episode is None:
            raise RuntimeError("step() called without reset()")
        if self._episode.done:
            raise RuntimeError("step() called on a terminated episode")

        ep = self._episode
        rc = self._reward_computer
        ep.step_count += 1
        self._state.step_count += 1

        tool_name = action.tool_name
        args = action.args

        # --- Route tool call ---
        try:
            if tool_name == "list_passengers":
                tool_result = tool_list_passengers(ep)
                reward, reason = rc.reward_for_info_call("list_passengers", ep)

            elif tool_name == "get_passenger_details":
                pid = args.get("passenger_id", "")
                already_booked = pid in ep.bookings
                tool_result = tool_get_passenger_details(ep, pid)
                if tool_result["status"] == "error":
                    reward, reason = rc.reward_for_failed_action(tool_result)
                elif already_booked:
                    reward, reason = rc.reward_for_info_call(
                        "get_passenger_details_booked", ep
                    )
                else:
                    reward, reason = rc.reward_for_info_call(
                        "get_passenger_details", ep
                    )

            elif tool_name == "list_alternative_flights":
                tool_result = tool_list_alternative_flights(ep)
                ep.flights_listed = True
                reward, reason = rc.reward_for_info_call(
                    "list_alternative_flights", ep
                )

            elif tool_name == "get_flight_details":
                fid = args.get("flight_id", "")
                tool_result = tool_get_flight_details(ep, fid)
                if tool_result["status"] == "error":
                    reward, reason = rc.reward_for_failed_action(tool_result)
                else:
                    reward, reason = rc.reward_for_info_call(
                        "get_flight_details", ep
                    )

            elif tool_name == "book_passenger":
                pid = args.get("passenger_id", "")
                fid = args.get("flight_id", "")
                cabin = args.get("cabin", "")
                tool_result = tool_book_passenger(ep, pid, fid, cabin)
                if tool_result["status"] == "success":
                    pax = ep.passengers.get(pid, {})
                    reward, reason = rc.reward_for_booking(
                        tool_result, pax, ep
                    )
                else:
                    reward, reason = rc.reward_for_failed_action(tool_result)

            elif tool_name == "book_group":
                gid = args.get("group_id", "")
                fid = args.get("flight_id", "")
                cabin_assignments = args.get("cabin_assignments", {})
                tool_result = tool_book_group(ep, gid, fid, cabin_assignments)
                if tool_result["status"] == "success":
                    group_pax = [
                        ep.passengers[pid]
                        for pid in ep.groups.get(gid, [])
                    ]
                    reward, reason = rc.reward_for_group_booking(
                        tool_result, group_pax, ep
                    )
                else:
                    reward, reason = rc.reward_for_failed_action(tool_result)

            elif tool_name == "finalize_plan":
                tool_result = tool_finalize_plan(ep)
                reward, reason = REWARD_FINALIZE, "Plan finalized"

            else:
                tool_result = {
                    "status": "error",
                    "message": f"Unknown tool: {tool_name!r}",
                }
                reward, reason = rc.reward_for_invalid_tool()

        except Exception as exc:
            tool_result = {"status": "error", "message": f"Internal error: {exc}"}
            reward, reason = rc.reward_for_invalid_tool()

        # --- Check termination ---
        all_booked = len(ep.bookings) >= len(ep.passengers)
        step_limit_reached = ep.step_count >= ep.max_steps
        done = bool(ep.done or all_booked or step_limit_reached)

        # --- Terminal grading ---
        if done:
            breakdown = rc.terminal_breakdown(
                ep.bookings, ep.passengers, ep.flights, ep.groups
            )
            grader = rc.grader_score(
                ep.bookings, ep.passengers, ep.flights, ep.groups
            )

            if step_limit_reached and not all_booked and not ep.done:
                reason += " | Episode timed out — not all passengers booked"
            elif all_booked and not ep.done:
                reason += " | All passengers booked — auto-finalized"
            else:
                reason += " | Episode complete"

            tool_result = {
                **(tool_result or {}),
                "terminal_breakdown": breakdown,
                "grader_score": grader,
            }
            ep.done = True

        ep.cumulative_reward += reward

        return self._build_observation(
            tool_result=tool_result,
            reward=reward,
            reward_reason=reason,
            done=done,
        )

    @property
    def state(self) -> FlightRebookingState:  # type: ignore[override]
        if self._episode is None:
            return FlightRebookingState()

        ep = self._episode
        n_booked = len(ep.bookings)

        return FlightRebookingState(
            episode_id=self._state.episode_id,
            step_count=ep.step_count,
            total_passengers=len(ep.passengers),
            passengers_booked=n_booked,
            passengers_remaining=len(ep.passengers) - n_booked,
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
    ) -> FlightRebookingObservation:
        ep = self._episode
        n_booked = len(ep.bookings)

        booked_summary = [
            {
                "passenger_id": pid,
                "flight_id": booking["flight_id"],
                "cabin": booking["cabin"],
            }
            for pid, booking in ep.bookings.items()
        ]

        # Only include flights snapshot if agent has called list_alternative_flights
        flights_snapshot = None
        if ep.flights_listed:
            flights_snapshot = [
                {
                    "flight_id": fid,
                    "departure_time": fl["departure_time"],
                    "arrival_time": fl["arrival_time"],
                    "cabin_availability": dict(ep.flight_availability[fid]),
                    "supports_ssr": fl["supports_ssr"],
                }
                for fid, fl in ep.flights.items()
            ]

        return FlightRebookingObservation(
            passengers_total=len(ep.passengers),
            passengers_booked=n_booked,
            passengers_remaining=len(ep.passengers) - n_booked,
            tool_result=tool_result,
            reward_reason=reward_reason,
            step_count=ep.step_count,
            max_steps=ep.max_steps,
            cumulative_reward=ep.cumulative_reward,
            booked_summary=booked_summary,
            flights_snapshot=flights_snapshot,
            done=done,
            reward=reward,
        )


# Import needed for finalize reward constant
from server.rewards import REWARD_FINALIZE  # noqa: E402
