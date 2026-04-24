# Flight Rebooking Environment — Claude Code Project Context

## Project Identity

This is an **OpenEnv-compliant RL environment** for a Meta hackathon ("Team Agentic Troop"). We are **migrating** from a seat-reassignment-after-aircraft-swap scenario to a **flight rebooking/reaccommodation after cancellation** scenario.

**Repo**: Python 3.10+ / FastAPI / OpenEnv-core  
**Deployed on**: HuggingFace Spaces (Docker)  
**Package name being renamed**: `seat_reassignment` → `flight_rebooking`

---

## Migration Summary

| Aspect | OLD (seat_reassignment) | NEW (flight_rebooking) |
|--------|------------------------|----------------------|
| Problem | Aircraft swap — reassign seats AC-1→AC-2 | Flight cancelled — rebook passengers onto alternative flights |
| Granularity | Seat-level (specific seats) | Inventory-level (cabin buckets on flights) |
| Tools | 3 (get_passenger_details, assign_seat, swap_seats) | 7 (list_passengers, get_passenger_details, list_alternative_flights, get_flight_details, book_passenger, book_group, finalize_plan) |
| Constraints | Cabin class, paid_window, paid_legroom | Priority tiers, groups (hard/soft), SSR flags (UM, WCHR, pet_cabin, pet_cargo), downstream deadlines, cabin matching |
| Reward | Per-step + terminal weighted score | 3-layer: step-shaped + end-of-episode settlement + hard-constraint penalties |
| Grader | cabin_score ± preference_score | 0.35×coverage + 0.15×cabin_match + 0.15×group_integrity + 0.15×deadline + 0.20×ssr_integrity |
| Difficulty | Data varies (8 vs 20 passengers, preference types) | Data varies (passenger count, constraint density, capacity scarcity) — same prompt all tiers |

---

## Architecture (Must Preserve)

The OpenEnv contract MUST be maintained:
- `server/app.py` — FastAPI app created via `create_app()` from openenv-core
- `server/environment.py` — `Environment` subclass with `reset(seed, task_id)`, `step(action)`, `state` property
- `models.py` — Pydantic models extending `Action`, `Observation`, `State` from openenv
- `client.py` — `EnvClient` WebSocket client
- `openenv.yaml` — spec metadata
- `Dockerfile` — multi-stage build with openenv-base
- `inference.py` — baseline inference script using OpenAI client

---

## Key Rules

1. **Never break the OpenEnv interface.** `reset()` returns Observation. `step()` takes Action, returns Observation. `state` returns State.
2. **All tool logic lives in `server/tools.py`.** Tools are pure functions taking state + args → dict.
3. **All reward logic lives in `server/rewards.py`.** Stateless `RewardComputer` class.
4. **Data files go in `data/{easy,medium,hard}/`.** JSON configs for flights, passengers. CSV is fine too.
5. **Models in `models.py` must inherit** from openenv base types (`Action`, `Observation`, `State`).
6. **The system prompt in `inference.py` is constant across all difficulty tiers.** Difficulty comes from data only.
7. **Grader score must be in (EPS, 1-EPS)** where EPS = 1e-4. Clamped, deterministic, reproducible.
8. **Tests must cover all 3 difficulty tiers** with known-good assignments that achieve grader_score ≈ 1.0.
9. **Step budget** = configurable per tier via data, not hardcoded.

---

## File Change Map

### Files to REWRITE (complete replacement):
- `server/environment.py` — new EpisodeState, new reset/step logic, flight-based inventory
- `server/tools.py` — 7 new tools replacing 3 old tools
- `server/rewards.py` — 3-layer reward system, new grader formula
- `models.py` — new Action/Observation/State with flight-booking fields
- `inference.py` — new system prompt, new tool calling logic
- `client.py` — updated model imports

### Files to UPDATE:
- `openenv.yaml` — rename to flight_rebooking
- `pyproject.toml` — rename package, update metadata
- `Dockerfile` — update CMD entry point, package name
- `__init__.py` — update imports/exports
- `server/__init__.py` — update docstring
- `README.md` — complete rewrite for new problem
- `.gitignore` / `.dockerignore` — minor updates if needed

### Files to CREATE:
- `data/easy/` — new JSON data files (flights, passengers, services)
- `data/medium/` — new JSON data files
- `data/hard/` — new JSON data files
- `data/easy/generate_data.py` (optional)
- `data/medium/generate_data.py` (optional)  
- `data/hard/generate_data.py` (optional)

### Files to REWRITE:
- `tests/test_environment.py` — all new test cases for flight rebooking
- `tests/test_rewards.py` — all new reward/grader tests

### Files to DELETE:
- `data/*/ac1_config.json`, `data/*/ac2_config.json` — aircraft configs no longer needed
- `data/*/seats_ac1.csv`, `data/*/seats_ac2.csv` — seat CSVs no longer needed
- `data/*/assignments.csv` — old assignment format
- `data/*/passengers.json` (medium) — replaced by new format
- `data/*/generate_data.py` (old versions)

---

## Naming Conventions

- Package: `flight_rebooking` (snake_case)
- Classes: `FlightRebookingAction`, `FlightRebookingObservation`, `FlightRebookingState`, `FlightRebookingEnvironment`, `FlightRebookingEnv`
- Tool functions: `tool_list_passengers`, `tool_get_passenger_details`, `tool_list_alternative_flights`, `tool_get_flight_details`, `tool_book_passenger`, `tool_book_group`, `tool_finalize_plan`

---

## Development Workflow

1. **Read `agent_docs/migration_spec.md`** before any implementation — it contains the full design.
2. **Read `agent_docs/architecture.md`** for data model schemas and file structure.
3. **Read `agent_docs/testing_strategy.md`** for test requirements.
4. Implement in this order: data → models → tools → rewards → environment → tests → inference → packaging.
5. Run `pytest tests/ -v` after each major module.
6. Run `uv run server` to verify the server starts.

---

## Tech Stack

- Python 3.10+
- FastAPI (via openenv-core)
- pandas (for data manipulation in environment)
- pydantic (models)
- openenv-core[core] >= 0.2.2
- openai (inference client)
- pytest (testing)
- Docker (deployment)
- uv (package management)
