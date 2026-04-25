"""
Pydantic models for the Flight Rebooking environment (plan-then-commit model).

FlightRebookingAction      — what the agent sends on every step()
FlightRebookingObservation — what the environment returns to the agent
FlightRebookingState       — episode-level metadata (returned by state property)
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class FlightRebookingAction(Action):
    """
    Agent action — a tool call with its named arguments.

    Valid tool_name values:
        get_full_manifest, get_flight_inventory, submit_plan, finalize_plan.
    Any other value is routed to the invalid-tool handler inside step().
    """

    tool_name: str = Field(
        ...,
        description=(
            "Tool to call: get_full_manifest | get_flight_inventory | "
            "submit_plan | finalize_plan"
        ),
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Named arguments for the tool (values may be str or dict)",
    )


class FlightRebookingObservation(Observation):
    """
    Observation returned to the agent after every reset() and step().

    Inherited from Observation (do not redeclare): done, reward, metadata.
    """

    # Counters
    passengers_total: int = Field(
        default=0,
        description="Total number of passengers needing rebooking",
    )
    passengers_booked: int = Field(
        default=0,
        description="Number of passengers accepted in the current plan",
    )
    passengers_remaining: int = Field(
        default=0,
        description="Passengers not yet accepted in any plan",
    )

    # Step feedback
    tool_result: Optional[dict] = Field(
        default=None,
        description="Result dict from the last tool call; None after reset()",
    )
    reward_reason: str = Field(
        default="Episode started",
        description="Human-readable explanation of the reward for the last step",
    )
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=0, description="Step limit for this episode")
    cumulative_reward: float = Field(
        default=0.0,
        description="Sum of all rewards received so far in this episode",
    )

    # Summary view
    booked_summary: List[dict] = Field(
        default_factory=list,
        description="List of accepted bookings: [{passenger_id, flight_id, cabin}]",
    )

    # Plan tracking
    plan_submitted: bool = Field(
        default=False,
        description="Whether a plan has been submitted",
    )


class FlightRebookingState(State):
    """
    Episode-level metadata returned by the state property.

    Inherited from State (do not redeclare): episode_id, step_count.
    """

    total_passengers: int = Field(default=0)
    passengers_booked: int = Field(default=0)
    passengers_remaining: int = Field(default=0)
    cumulative_reward: float = Field(default=0.0)
    is_complete: bool = Field(default=False)
    plan_submitted: bool = Field(default=False)
