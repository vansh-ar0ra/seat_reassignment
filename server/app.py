"""
FastAPI application for the Airline Reassignment Environment.

Exposes the AirlineReassignmentEnvironment over HTTP and WebSocket endpoints
compatible with the OpenEnv EnvClient.

Endpoints (all created automatically by create_app):
  POST /reset   — start a new episode
  POST /step    — execute one tool call
  GET  /state   — current episode metadata
  GET  /schema  — action/observation schemas
  WS   /ws      — WebSocket for persistent sessions
  GET  /health  — liveness check
  GET  /docs    — auto-generated API docs

Usage:
  uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    from openenv.core.env_server import create_fastapi_app as create_app  # type: ignore

try:
    from ..models import AirlineReassignmentAction, AirlineReassignmentObservation
    from .environment import AirlineReassignmentEnvironment
except ImportError:
    from models import AirlineReassignmentAction, AirlineReassignmentObservation
    from server.environment import AirlineReassignmentEnvironment


app = create_app(
    AirlineReassignmentEnvironment,
    AirlineReassignmentAction,
    AirlineReassignmentObservation,
    env_name="airline_reassignment",
    max_concurrent_envs=int(os.environ.get("MAX_ENVS", "4")),
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
