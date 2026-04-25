# Flight Rebooking Environment — Claude Code Project Context

## Project Identity

This is an **OpenEnv-compliant RL environment** for a Meta hackathon ("Team Agentic Troop"). The environment simulates **flight rebooking/reaccommodation after cancellation** — an agent must rebook passengers onto alternative flights respecting complex, interacting constraints.

**Repo**: Python 3.10+ / FastAPI / OpenEnv-core  
**Deployed on**: HuggingFace Spaces (Docker)  
**Package name**: `flight_rebooking` (directory still named `seat_reassignment`)

---

## Current Architecture

### Core Components

| File | Purpose |
|------|---------|
| `server/environment.py` | `FlightRebookingEnvironment` — OpenEnv `Environment` subclass with `reset()`, `step()`, `state` |
| `server/tools.py` | 8 tool functions (pure: state + args → dict) |
| `server/rewards.py` | `RewardComputer` — 3-layer reward system + 7-component grader |
| `models.py` | Pydantic models: `FlightRebookingAction`, `FlightRebookingObservation`, `FlightRebookingState` |
| `client.py` | `FlightRebookingEnv` — WebSocket client wrapping `EnvClient` |
| `inference.py` | Baseline LLM inference loop using OpenAI-compatible client |
| `data/generate.py` | Procedural data generator (seed-based, difficulty-scaled) |
| `server/app.py` | FastAPI app via `create_app()` from openenv-core |
| `openenv.yaml` | Spec metadata |

### Tools (8 total)

| # | Tool | Purpose |
|---|------|---------|
| 1 | `list_passengers` | Survey all passengers (lightweight summary) |
| 2 | `get_passenger_details` | Full record for one passenger |
| 3 | `list_alternative_flights` | All active flights with availability |
| 4 | `get_flight_details` | Details for one flight |
| 5 | `book_passenger` | Book one passenger onto a flight/cabin |
| 6 | `book_group` | Book entire group atomically (same flight) |
| 7 | `unbook_passenger` | Undo a booking — frees seat, reverses cost |
| 8 | `finalize_plan` | End episode, trigger grading |

### Grader Components (7, weights sum to 1.0)

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Coverage | 0.25 | Fraction of passengers booked |
| Cabin match | 0.15 | Priority-weighted same-cabin rate |
| Group integrity | 0.12 | Groups kept on same flight/cabin |
| Deadline | 0.13 | Deadline-bearing passengers arriving on time |
| SSR integrity | 0.15 | SSR requirements met by booked flight |
| Cost efficiency | 0.10 | Budget adherence + per-passenger cost |
| Loyalty compliance | 0.10 | Gold/silver members not downgraded |

Hard-constraint violations (SSR mismatch, hard group split) incur an additional 0.15 penalty each.

### Observation Fields

Standard: `passengers_total`, `passengers_booked`, `passengers_remaining`, `tool_result`, `reward_reason`, `step_count`, `max_steps`, `cumulative_reward`, `booked_summary`, `flights_snapshot`, `done`, `reward`

Added in complexity update:
- `reward_breakdown` — per-component deltas (coverage, cabin_match, group, deadline, ssr, cost, loyalty, opportunity_cost)
- `events` — mid-episode events that fired this step
- `total_cost` — accumulated cost (upgrades + compensation)
- `compensation_budget` — budget ceiling for the episode

### State Fields

`episode_id`, `step_count`, `total_passengers`, `passengers_booked`, `passengers_remaining`, `cumulative_reward`, `is_complete`, `total_cost`, `compensation_budget_remaining`

---

## Complexity Features

### Mid-Episode Events
Stochastic events fire at specific steps (procedural generation only, `events_enabled: true`):
- **capacity_change** — cabin seats added/removed on a flight
- **new_passenger** — missed-connection passenger added
- **ssr_equipment_failure** — flight loses SSR support
- **deadline_shift** — passenger's connection deadline changes
- **secondary_cancellation** — another flight cancels, unbooks affected passengers

### Cost Tracking
- Upgrade costs: economy→business = $800, economy→premium_economy = $200, etc.
- Downgrade compensation: business→economy = $700, etc.
- Loyalty entitlements: gold gets lounge ($40) + meal ($25), silver gets meal ($25)
- `compensation_budget` ceiling per episode; budget overrun penalized by grader

### Progressive Difficulty
- `_reward_scale = max(0.6, 1.0 - 0.4 * difficulty)` — positive rewards shrink at higher difficulty
- Penalties are unaffected by difficulty
- Step budgets tighten: ~2.5 steps/pax (easy) → ~2.0 steps/pax (hard)

### Procedural Generation
- `data/generate.py` generates passengers, flights, config, and events from a seed
- Difficulty 0.0–1.0 controls: passenger count (8–45), flight count (3–8), SSR density, group density, surplus ratio, step budget
- Adversarial scenarios: greedy traps, distractor flights, priority inversion, Pareto conflicts

### Decomposed Rewards
Each booking step returns a `reward_breakdown` dict with 8 component deltas, plus opportunity cost when scarce resources are consumed.

---

## Data Files

### Static data: `data/{easy,medium,hard}/`
Each tier directory contains:
- `config.json` — `task_id`, `max_steps`, `compensation_budget`, `difficulty`, `events_enabled`
- `passengers.json` — array of passenger records with `loyalty_status`
- `flights.json` — array of flight records with `cabin_availability`, `supports_ssr`

### Step budgets (static tiers)
| Tier | Passengers | Max Steps | Steps/Pax | Difficulty | Budget |
|------|-----------|-----------|-----------|------------|--------|
| easy | 8 | 20 | 2.5 | 0.2 | $5,000 |
| medium | 15 | 35 | 2.3 | 0.5 | $4,000 |
| hard | 25 | 55 | 2.2 | 0.8 | $5,000 |

---

## Key Rules

1. **Never break the OpenEnv interface.** `reset()` returns Observation. `step()` takes Action, returns Observation. `state` returns State.
2. **All tool logic lives in `server/tools.py`.** Tools are pure functions taking state + args → dict.
3. **All reward logic lives in `server/rewards.py`.** `RewardComputer` class.
4. **Data files go in `data/{easy,medium,hard}/`.** JSON configs for flights, passengers.
5. **Models in `models.py` must inherit** from openenv base types (`Action`, `Observation`, `State`).
6. **The system prompt in `inference.py` is constant across all difficulty tiers.** Difficulty comes from data only.
7. **Grader score must be in (EPS, 1-EPS)** where EPS = 1e-4. Clamped, deterministic, reproducible.
8. **Tests must cover all 3 difficulty tiers** with known-good assignments that achieve grader_score > 0.90.
9. **Step budget** = configurable per tier via data, not hardcoded.
10. **Backward compatibility**: `_load_static()` adds defaults for `loyalty_status`, `compensation_budget`, `difficulty`, `events_enabled` when missing from old data files.

---

## Naming Conventions

- Package: `flight_rebooking` (snake_case)
- Classes: `FlightRebookingAction`, `FlightRebookingObservation`, `FlightRebookingState`, `FlightRebookingEnvironment`, `FlightRebookingEnv`
- Tool functions: `tool_list_passengers`, `tool_get_passenger_details`, `tool_list_alternative_flights`, `tool_get_flight_details`, `tool_book_passenger`, `tool_book_group`, `tool_unbook_passenger`, `tool_finalize_plan`

---

## Development Workflow

1. **Read `agent_docs/migration_spec.md`** before any implementation — it contains the full design.
2. **Read `agent_docs/architecture.md`** for data model schemas and file structure.
3. **Read `agent_docs/testing_strategy.md`** for test requirements.
4. Implement in this order: data → models → tools → rewards → environment → tests → inference → packaging.
5. Run tests: `cd` to project root, then `.venv/Scripts/python.exe -m pytest tests/ -v`
6. Run server: `uv run server` to verify the server starts.

---

## Tech Stack

- Python 3.10+
- FastAPI (via openenv-core)
- pydantic (models)
- openenv-core[core] >= 0.2.3
- openai (inference client)
- pytest (testing)
- Docker (deployment)
- uv (package management)
- Virtual environment at `.venv/` (has openenv-core installed)
