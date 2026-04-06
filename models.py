"""
Pydantic models for the Seat Swap environment.

SeatSwapAction    — what the agent sends on every step()
SeatSwapObservation — what the environment returns to the agent
SeatSwapState     — episode-level metadata (returned by state property)
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SeatSwapAction(Action):
    """
    Agent action — a tool call with its named arguments.

    Valid tool_name values: "get_passenger_details", "assign_seat", "swap_seats".
    Any other value is routed to the invalid-tool handler inside step().
    """

    tool_name: str = Field(
        ...,
        description="Tool to call: get_passenger_details | assign_seat | swap_seats",
    )
    args: Dict[str, str] = Field(
        default_factory=dict,
        description="Named arguments for the tool (all values are strings)",
    )


class SeatSwapObservation(Observation):
    """
    Observation returned to the agent after every reset() and step().

    Inherited from Observation (do not redeclare): done, reward, metadata.
    """

    # Static aircraft layout context (never changes during an episode)
    ac1_layout: dict = Field(
        default_factory=dict,
        description="Full AC-1 seat configuration including cabin/seat_type for each seat",
    )
    ac2_layout: dict = Field(
        default_factory=dict,
        description="Full AC-2 seat configuration including cabin/seat_type for each seat",
    )

    # Dynamic assignment state
    ac1_seats_occupied: List[str] = Field(
        default_factory=list,
        description="AC-1 seat IDs whose passengers have NOT yet been moved to AC-2",
    )
    ac2_seats_occupied: Dict[str, str] = Field(
        default_factory=dict,
        description="AC-2 seat_id → passenger_id for currently occupied AC-2 seats",
    )
    ac2_seats_available: List[str] = Field(
        default_factory=list,
        description="AC-2 seat IDs that are still empty",
    )

    # Counters
    passengers_remaining: int = Field(
        default=0,
        description="Passengers not yet assigned to AC-2",
    )
    passengers_total: int = Field(
        default=0,
        description="Total passenger count (always 20 in the current dataset)",
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
    grader_score: Optional[float] = Field(
        default=None,
        description="Current 0-1 quality score; None until episode ends",
    )


class SeatSwapState(State):
    """
    Episode-level metadata returned by the state property.

    Inherited from State (do not redeclare): episode_id, step_count.
    """

    total_passengers: int = Field(default=0)
    passengers_assigned: int = Field(default=0)
    passengers_remaining: int = Field(default=0)
    cumulative_reward: float = Field(default=0.0)
    is_complete: bool = Field(default=False)
