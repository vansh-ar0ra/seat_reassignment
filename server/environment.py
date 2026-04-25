"""
Core environment implementation for the Flight Rebooking task (plan-then-commit model).

A flight has been cancelled. The agent must rebook passengers onto
alternative flights using 4 tools:
  - get_full_manifest       -> get all passenger details in one call
  - get_flight_inventory    -> get all flights with availability
  - submit_plan             -> submit a complete rebooking plan (one shot)
  - finalize_plan           -> lock in the plan and trigger grading

An episode ends when finalize_plan is called or the step limit is reached.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
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
        tool_get_full_manifest,
        tool_get_flight_inventory,
        tool_submit_plan,
        tool_finalize_plan,
    )
    from .rewards import RewardComputer
    from .debug import RunDebugger
except ImportError:
    from models import (
        FlightRebookingAction,
        FlightRebookingObservation,
        FlightRebookingState,
    )
    from server.tools import (
        tool_get_full_manifest,
        tool_get_flight_inventory,
        tool_submit_plan,
        tool_finalize_plan,
    )
    from server.rewards import RewardComputer
    from server.debug import RunDebugger


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
    initial_availability: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Plan tracking
    plan_submitted: bool = False
    last_plan_preview: float = 0.0

    # Info-call tracking (tool_name -> call count)
    info_call_counts: Dict[str, int] = field(default_factory=dict)

    # Episode metadata
    task_id: str = ""
    step_count: int = 0
    max_steps: int = 5
    cumulative_reward: float = 0.0
    done: bool = False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FlightRebookingEnvironment(Environment):
    """
    OpenEnv environment for airline flight rebooking after cancellation.

    Plan-then-commit model: the agent gathers info, submits a complete
    rebooking plan (one shot, no revisions), then finalizes.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, debug: bool = True, runs_dir: Optional[str] = None):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode: Optional[EpisodeState] = None
        self._reward_computer: Optional[RewardComputer] = None
        self._data_dir = Path(__file__).resolve().parent.parent / "data"
        self._debugger = RunDebugger(runs_dir=runs_dir, enabled=debug)

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
        initial_availability = copy.deepcopy(flight_availability)

        max_steps = config.get("max_steps", 5)

        self._episode = EpisodeState(
            passengers=passengers,
            flights=flights,
            groups=groups,
            config=config,
            bookings={},
            flight_availability=flight_availability,
            initial_availability=initial_availability,
            plan_submitted=False,
            last_plan_preview=0.0,
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

        self._debugger.start(task_id, episode_id)

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
            if tool_name == "get_full_manifest":
                tool_result = tool_get_full_manifest(ep)
                repeated = ep.info_call_counts.get(tool_name, 0) > 0
                ep.info_call_counts[tool_name] = ep.info_call_counts.get(tool_name, 0) + 1
                reward, reason = rc.reward_for_info_call(repeated=repeated)

            elif tool_name == "get_flight_inventory":
                tool_result = tool_get_flight_inventory(ep)
                repeated = ep.info_call_counts.get(tool_name, 0) > 0
                ep.info_call_counts[tool_name] = ep.info_call_counts.get(tool_name, 0) + 1
                reward, reason = rc.reward_for_info_call(repeated=repeated)

            elif tool_name == "submit_plan":
                # Accept both formats:
                #   {"assignments": {"PAX-001": {...}, ...}}   (canonical)
                #   {"PAX-001": {...}, ...}                    (flat — LLMs often omit the wrapper)
                if "assignments" in args:
                    assignments = args["assignments"]
                else:
                    assignments = args
                if ep.plan_submitted:
                    tool_result = tool_submit_plan(ep, assignments, rc)
                    reward, reason = rc.reward_for_duplicate_submit()
                else:
                    tool_result = tool_submit_plan(ep, assignments, rc)
                    if tool_result["status"] == "success":
                        preview = tool_result["plan_score_preview"]
                        reward, reason = rc.reward_for_plan_submission(preview)
                    else:
                        reward, reason = rc.reward_for_info_call()  # unexpected error

            elif tool_name == "finalize_plan":
                tool_result = tool_finalize_plan(ep)
                reward, reason = rc.reward_for_finalize(ep.plan_submitted)

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
        step_limit_reached = ep.step_count >= ep.max_steps
        done = bool(ep.done or step_limit_reached)

        # --- Terminal grading ---
        if done:
            breakdown = rc.terminal_breakdown(
                ep.bookings, ep.passengers, ep.flights, ep.groups
            )
            grader = rc.grader_score(
                ep.bookings, ep.passengers, ep.flights, ep.groups
            )

            if step_limit_reached and not ep.done:
                reason += " | Episode timed out"
            else:
                reason += " | Episode complete"

            tool_result = {
                **(tool_result or {}),
                "terminal_breakdown": breakdown,
                "grader_score": grader,
            }
            ep.done = True

            self._debugger.log_terminal(
                bookings=ep.bookings,
                passengers=ep.passengers,
                flights=ep.flights,
                groups=ep.groups,
                grader_score=grader,
                breakdown=breakdown,
            )

        ep.cumulative_reward += reward

        self._debugger.log_step(
            step=ep.step_count,
            tool_name=tool_name,
            args=args,
            tool_result=tool_result,
            reward=reward,
            reward_reason=reason,
            cumulative_reward=ep.cumulative_reward,
            passengers_booked=len(ep.bookings),
            passengers_total=len(ep.passengers),
            done=done,
        )

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
            plan_submitted=ep.plan_submitted,
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
            plan_submitted=ep.plan_submitted,
            done=done,
            reward=reward,
        )
