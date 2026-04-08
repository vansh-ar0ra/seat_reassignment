"""Seat Reassignment Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import SeatReassignmentAction, SeatReassignmentObservation, SeatReassignmentState
except ImportError:
    from models import SeatReassignmentAction, SeatReassignmentObservation, SeatReassignmentState


class SeatReassignmentEnv(EnvClient[SeatReassignmentAction, SeatReassignmentObservation, SeatReassignmentState]):
    """
    WebSocket client for the Seat Reassignment Environment.

    Example:
        >>> with SeatReassignmentEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     obs = result.observation
        ...     print(obs.passengers_remaining)  # 20
        ...
        ...     action = SeatReassignmentAction(
        ...         tool_name="get_passenger_details",
        ...         args={"seat_id": "1A"},
        ...     )
        ...     result = env.step(action)
        ...     print(result.observation.tool_result)
    """

    def _step_payload(self, action: SeatReassignmentAction) -> Dict:
        return {"tool_name": action.tool_name, "args": action.args}

    def _parse_result(self, payload: Dict) -> StepResult[SeatReassignmentObservation]:
        obs_data = payload.get("observation", payload)
        obs = SeatReassignmentObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SeatReassignmentState:
        return SeatReassignmentState(**payload)
