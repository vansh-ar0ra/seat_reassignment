# Flight Rebooking Environment -- Project Documentation

## Table of Contents

1. [Problem Statement & Use Case](#1-problem-statement--use-case)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Model](#3-data-model)
4. [Tools (Agent Actions)](#4-tools-agent-actions)
5. [Environment Lifecycle: Reset & Step](#5-environment-lifecycle-reset--step)
6. [Reward System](#6-reward-system)
7. [Grader (Terminal Scoring)](#7-grader-terminal-scoring)
8. [Difficulty Tiers & Data Design](#8-difficulty-tiers--data-design)
9. [Inference Pipeline](#9-inference-pipeline)
10. [Deployment & Packaging](#10-deployment--packaging)
11. [Testing Strategy](#11-testing-strategy)

---

## 1. Problem Statement & Use Case

### The Scenario

A scheduled airline flight (FL-100) has been **cancelled**. All passengers on that flight must be rebooked onto alternative flights to the same destination (DEL). An AI agent operates as the airline operations controller, making rebooking decisions one tool call at a time.

### Why This Is Hard

This is not a simple assignment problem. The agent must juggle multiple competing constraints simultaneously:

- **SSR compatibility**: Passengers with special service requirements (unaccompanied minors, wheelchair users, travelers with pets) can only go on flights that support those services. Not every flight supports every SSR.
- **Group integrity**: Families and travel groups must stay together. "Hard" groups **must** be on the same flight (non-negotiable). "Soft" groups **should** be kept together but can be split if necessary.
- **Downstream deadlines**: Some passengers have connecting flights. Their rebooking must land before their connection departs.
- **Cabin matching**: Passengers paid for a specific cabin class (economy, premium economy, business). Downgrading them is bad; matching is good; upgrading is acceptable.
- **Priority tiers**: Frequent flyers and premium passengers (tier 1) should get better outcomes than occasional travelers (tier 5) when trade-offs are required.
- **Capacity scarcity**: Alternative flights have limited seats per cabin. The agent can't just dump everyone on the first available flight.

### The RL Framing

This is an **OpenEnv-compliant reinforcement learning environment**. The agent interacts through a tool-calling interface:

```
Agent sends: {"tool_name": "book_passenger", "args": {"passenger_id": "PAX-001", ...}}
Environment returns: observation with tool result, reward, booking status, done flag
```

The agent receives **step-level rewards** for good/bad decisions (shaping signal) and a **terminal grader score** (0 to 1) that evaluates the overall quality of the rebooking plan.

---

## 2. Architecture Overview

### System Components

```
                    +--------------------------+
                    |      inference.py         |
                    |   (LLM agent loop)        |
                    +--------+---------+--------+
                             |         ^
                     action  |         | observation
                             v         |
                    +--------+---------+--------+
                    |       client.py            |
                    |  (WebSocket EnvClient)     |
                    +--------+---------+--------+
                             |         ^
                     JSON    |         | JSON
                             v         |
                    +--------+---------+--------+
                    |      server/app.py         |
                    |  (FastAPI via create_app)  |
                    +--------+---------+--------+
                             |         ^
                             v         |
                    +--------+---------+--------+
                    | server/environment.py      |
                    | (FlightRebookingEnvironment)|
                    +---+------+------+---------+
                        |      |      |
           +------------+  +---+---+  +----------+
           |               |       |             |
    server/tools.py  server/rewards.py  data/{tier}/
    (7 tool fns)     (RewardComputer)   (JSON configs)
```

### Key Files

| File | Role |
|------|------|
| `models.py` | Pydantic models: `FlightRebookingAction`, `FlightRebookingObservation`, `FlightRebookingState` -- inheriting from OpenEnv base types |
| `server/environment.py` | Core `FlightRebookingEnvironment` class with `reset()`, `step()`, `state` -- the OpenEnv contract |
| `server/tools.py` | 7 pure functions implementing each tool. Take episode state + args, return result dicts |
| `server/rewards.py` | `RewardComputer` class: step-level rewards + terminal grader score |
| `server/app.py` | FastAPI app created via `create_app()` from openenv-core. Exposes REST + WebSocket endpoints |
| `client.py` | `FlightRebookingEnv` -- typed WebSocket client for remote interaction |
| `inference.py` | Baseline agent: LLM loop using OpenAI client, system prompt, tool-call parsing |
| `data/{easy,medium,hard}/` | JSON data files defining passengers, flights, and config per difficulty tier |

### OpenEnv Contract

The environment **must** expose:

- `reset(seed, task_id) -> Observation` -- initialize a new episode
- `step(action) -> Observation` -- execute one tool call
- `state -> State` -- current episode metadata

These are the only interface points. Everything else is internal.

---

## 3. Data Model

### 3.1 Passenger Record

Each passenger is a JSON object loaded from `data/{tier}/passengers.json`:

```json
{
    "passenger_id": "PAX-M001",
    "name": "Kabir Malhotra",
    "priority_tier": 1,
    "original_cabin": "business",
    "group_id": null,
    "group_integrity": null,
    "group_size": null,
    "ssr_flags": ["UM"],
    "downstream_deadline": null,
    "paid_window": true,
    "paid_legroom": true
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `passenger_id` | string | Unique ID (e.g., `PAX-E001`, `PAX-M003`, `PAX-H017`) |
| `priority_tier` | int 1-5 | 1 = highest priority (frequent flyer), 5 = lowest |
| `original_cabin` | string | Cabin class on the cancelled flight: `economy`, `premium_economy`, `business` |
| `group_id` | string or null | Group identifier if traveling with others |
| `group_integrity` | `"hard"` / `"soft"` / null | `hard` = must stay together (same flight), `soft` = should try to stay together |
| `group_size` | int or null | Number of members in the group |
| `ssr_flags` | list of strings | Special service requirements: `UM` (unaccompanied minor), `WCHR` (wheelchair), `pet_cabin`, `pet_cargo` |
| `downstream_deadline` | `"HH:MM"` or null | Latest acceptable arrival time (connecting flight) |
| `paid_window` | bool | Paid for a window seat preference |
| `paid_legroom` | bool | Paid for extra legroom preference |

### 3.2 Flight Record

Each alternative flight is loaded from `data/{tier}/flights.json`:

```json
{
    "flight_id": "FL-201",
    "departure_time": "09:00",
    "arrival_time": "12:15",
    "cabin_availability": {
        "economy": 8,
        "premium_economy": 4,
        "business": 3
    },
    "seat_features": {
        "economy": {"window": 3, "legroom": 2},
        "premium_economy": {"window": 2, "legroom": 2},
        "business": {"window": 1, "legroom": 3}
    },
    "supports_ssr": ["UM", "WCHR"]
}
```

| Field | Type | Meaning |
|-------|------|---------|
| `flight_id` | string | Unique flight identifier |
| `departure_time` | `"HH:MM"` | Departure time |
| `arrival_time` | `"HH:MM"` | Arrival time -- used for deadline checks |
| `cabin_availability` | dict | Seats available per cabin class. **Decremented** as bookings are made |
| `seat_features` | dict | Window and legroom seats available per cabin (for preference satisfaction) |
| `supports_ssr` | list of strings | Which special service requirements this flight can accommodate |

### 3.3 Config

Per-tier metadata in `data/{tier}/config.json`:

```json
{
    "task_id": "medium",
    "max_steps": 60,
    "cancelled_flight": "FL-100",
    "destination": "DEL"
}
```

### 3.4 Internal Episode State

At runtime, the environment maintains an `EpisodeState` dataclass (not exposed to the agent):

```python
@dataclass
class EpisodeState:
    passengers: Dict[str, dict]          # passenger_id -> full record
    flights: Dict[str, dict]             # flight_id -> full record
    groups: Dict[str, List[str]]         # group_id -> [passenger_ids]
    config: dict

    bookings: Dict[str, dict]            # passenger_id -> {flight_id, cabin}
    flight_availability: Dict[str, Dict[str, int]]  # flight_id -> {cabin: count}

    info_calls: Dict[str, int]           # tool_name -> call count
    last_booking_step: int               # for churn detection
    passenger_details_fetched: Set[str]  # which passengers have been inspected
    flights_listed: bool                 # whether list_alternative_flights called

    task_id: str
    step_count: int
    max_steps: int
    cumulative_reward: float
    done: bool
```

---

## 4. Tools (Agent Actions)

The agent has 7 tools available. Each step, it calls exactly one tool. Tools are pure functions in `server/tools.py` that take the episode state and return a result dict.

### 4.1 Information-Gathering Tools

#### `list_passengers()`

**Purpose**: Survey all passengers at a glance.

**Returns**: Lightweight summary per passenger -- ID, priority tier, group ID, boolean flags for SSR/deadline/booked status. Does NOT include full details like original cabin or exact SSR flags.

**Why**: Lets the agent plan its approach without drowning in detail. Should be called first.

#### `get_passenger_details(passenger_id)`

**Purpose**: Get the full record for one passenger.

**Returns**: Everything -- original cabin, exact SSR flags, group membership details, deadline, priority tier.

**Validation**: Returns error if passenger doesn't exist.

**Note**: Also shows current booking if the passenger is already booked.

#### `list_alternative_flights()`

**Purpose**: See all available flights with current seat counts.

**Returns**: Every flight with departure/arrival times, per-cabin availability (live -- reflects bookings made so far), and SSR support.

**Side effect**: Populates the `flights_snapshot` field in the observation (visible on subsequent steps).

#### `get_flight_details(flight_id)`

**Purpose**: Inspect one specific flight in detail.

**Returns**: Full flight record with current availability.

**Validation**: Returns error if flight doesn't exist.

### 4.2 Commitment Tools

#### `book_passenger(passenger_id, flight_id, cabin)`

**Purpose**: Book a single passenger onto a specific flight and cabin.

**Validation chain** (in order):
1. Passenger exists
2. Passenger is not already booked
3. Flight exists
4. Cabin is valid (`economy` | `premium_economy` | `business`)
5. Cabin has availability > 0 on that flight
6. Flight supports ALL of the passenger's SSR flags
7. If passenger has a deadline, the flight arrives before it
8. If passenger is in a hard group, a warning is issued (should use `book_group`)

**On success**: Decrements cabin availability, records the booking, reports cabin match status.

**On failure**: Returns error with specific reason. Nothing is changed.

#### `book_group(group_id, flight_id, cabin_assignments)`

**Purpose**: Book an entire group onto one flight **atomically** (all or none).

`cabin_assignments` is a dict mapping each member's passenger ID to their cabin assignment. This allows split-cabin bookings (e.g., some members in economy, others in premium_economy) on the same flight.

**Validation chain** (in order):
1. Group exists
2. All group members present in `cabin_assignments`, and none already booked
3. Flight exists
4. All cabin assignments are valid
5. Flight has sufficient capacity for ALL members in their respective cabins
6. Flight supports SSR flags of ALL group members
7. All members' deadlines (if any) are met by the flight's arrival time

**On success**: All members booked atomically. Availability decremented for each.

**On failure**: Nothing changes. Returns error explaining what went wrong.

**Why atomic**: Prevents partial group bookings that would violate group integrity. If 3 members need economy seats but only 2 are available, none get booked -- the agent must find a different flight or cabin arrangement.

#### `finalize_plan()`

**Purpose**: Signal that the agent is done. Triggers terminal grading.

**Returns**: Status dict. The environment's `step()` method computes and attaches the grader score and breakdown to the final observation.

### 4.3 Tool Summary Table

| # | Tool | Args | Mutates State? | Typical Use |
|---|------|------|---------------|-------------|
| 1 | `list_passengers` | none | no | First call -- survey the problem |
| 2 | `get_passenger_details` | `passenger_id` | no | Before booking -- check constraints |
| 3 | `list_alternative_flights` | none | no | Survey capacity, refresh after bookings |
| 4 | `get_flight_details` | `flight_id` | no | Spot-check one flight's availability |
| 5 | `book_passenger` | `passenger_id, flight_id, cabin` | **yes** | Commit one passenger to a flight |
| 6 | `book_group` | `group_id, flight_id, cabin_assignments` | **yes** | Commit an entire group atomically |
| 7 | `finalize_plan` | none | **yes** (ends episode) | Signal completion, trigger grading |

---

## 5. Environment Lifecycle: Reset & Step

### 5.1 `reset(seed, task_id)`

Called to initialize a new episode. `task_id` is one of `"easy"`, `"medium"`, `"hard"`.

**What happens**:

1. **Resolve data directory**: `data/{task_id}/` -- raises `ValueError` if directory doesn't exist
2. **Load files**: `config.json`, `passengers.json`, `flights.json`
3. **Build lookup structures**:
   - `passengers` dict: passenger_id -> full record
   - `flights` dict: flight_id -> full record
   - `groups` dict: group_id -> [member passenger_ids] (derived from passenger records)
   - `flight_availability` dict: deep copy of each flight's `cabin_availability` (so bookings decrement the copy, not the source)
4. **Initialize EpisodeState** with all tracking fields zeroed out
5. **Create RewardComputer** with total passenger count and max steps
6. **Return initial observation**: done=False, reward=0.0, no tool result, counters populated

### 5.2 `step(action)`

Called with a `FlightRebookingAction` containing `tool_name` and `args`.

**What happens**:

1. **Validate state**: Raise `RuntimeError` if `reset()` hasn't been called or episode is already done
2. **Increment step counter**
3. **Route tool call**: Match `tool_name` to one of the 7 tool functions. Pass episode state and args. Catch exceptions gracefully.
4. **Compute step reward**: Based on tool outcome (info call, successful booking, failed booking, invalid tool)
5. **Check termination conditions**:
   - `finalize_plan` was called (sets `ep.done = True`)
   - All passengers are booked (`len(bookings) >= len(passengers)`)
   - Step limit reached (`step_count >= max_steps`)
6. **If terminal**: Compute the grader score and breakdown, attach to tool result
7. **Accumulate reward**: Add step reward to `cumulative_reward`
8. **Build and return observation**

### 5.3 Observation Structure

The observation (`FlightRebookingObservation`) returned after every `reset()` and `step()`:

```python
class FlightRebookingObservation(Observation):
    passengers_total: int          # Total passengers needing rebooking
    passengers_booked: int         # Successfully booked so far
    passengers_remaining: int      # Not yet booked

    tool_result: Optional[dict]    # Result from the last tool call (None after reset)
    reward_reason: str             # Human-readable explanation of the reward
    step_count: int                # Current step number
    max_steps: int                 # Step budget for this episode
    cumulative_reward: float       # Running total of all rewards

    booked_summary: List[dict]     # [{passenger_id, flight_id, cabin}, ...]
    flights_snapshot: Optional[List[dict]]  # Populated only after list_alternative_flights

    done: bool                     # From parent: True if episode is over
    reward: float                  # From parent: reward for this step
```

**Design decision**: The observation is intentionally lightweight. It does NOT dump the full passenger manifest or flight list every step. The agent must use `list_passengers` / `list_alternative_flights` to discover that data. This:
- Keeps LLM context manageable
- Mirrors real-world API patterns
- Rewards agents that gather information strategically

### 5.4 Episode Termination

An episode ends when **any** of these conditions is met:

| Condition | Trigger | Note |
|-----------|---------|------|
| `finalize_plan()` called | `ep.done = True` set by tool | Agent's explicit signal |
| All passengers booked | `len(bookings) >= len(passengers)` | Auto-finalize |
| Step limit reached | `step_count >= max_steps` | Timeout -- unbooked passengers count as failures |

On termination, the `tool_result` in the final observation includes:
- `terminal_breakdown`: Dict with all grader sub-scores
- `grader_score`: The final 0-1 score

---

## 6. Reward System

The reward system has two layers: **step-level rewards** (immediate feedback per action) and a **terminal grader score** (end-of-episode evaluation).

### 6.1 Step-Level Rewards

These provide shaping signal to guide the agent's behavior during the episode.

#### Information Tool Rewards

| Tool Call | Reward | Rationale |
|-----------|--------|-----------|
| `list_passengers` (first call) | **+0.02** | Good planning -- the agent is surveying the problem |
| `list_passengers` (3rd+ call, no bookings since) | **-0.01** | Churn -- the agent is spinning its wheels |
| `get_passenger_details` (unbooked passenger) | **+0.02** | Useful -- inspecting before booking |
| `get_passenger_details` (already-booked passenger) | **-0.01** | Wasted step -- passenger is already handled |
| `list_alternative_flights` | **+0.01** | Always useful since inventory changes after bookings |
| `get_flight_details` | **+0.01** | Precise checking before committing |

#### Booking Outcome Rewards

All booking rewards are **multiplied by the passenger's priority weight**:

| Priority Tier | Weight |
|--------------|--------|
| Tier 1 (highest) | 1.5 |
| Tier 2 | 1.3 |
| Tier 3 | 1.0 |
| Tier 4 | 0.8 |
| Tier 5 (lowest) | 0.6 |

| Outcome | Base Reward | Example (Tier 1) |
|---------|-------------|-------------------|
| Same cabin booking | **+0.30** x weight | +0.45 |
| Cabin upgrade | **+0.10** x weight | +0.15 |
| Cabin downgrade | **-0.02** x weight | -0.03 |
| Deadline met (bonus, additive) | **+0.05** x weight | +0.075 |
| Failed booking (rejected by env) | **-0.50** | -0.50 (flat) |
| Invalid/unknown tool | **-0.20** | -0.20 (flat) |

#### Finalize Reward

`finalize_plan` itself carries **0.0** step reward. The grading happens in the terminal breakdown.

### 6.2 Group Booking Rewards

For `book_group`, the reward is the **sum of per-member rewards**. Each member is evaluated individually for cabin match/upgrade/downgrade, then summed. If all members end up in the same cabin on the same flight, the reason string reflects "all same cabin"; otherwise it notes "split cabin on same flight".

---

## 7. Grader (Terminal Scoring)

At episode end, the `RewardComputer.grader_score()` method computes a single score in **(EPS, 1-EPS)** where EPS = 1e-4. This is the score used for hackathon evaluation.

### 7.1 Component Weights

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| **coverage_score** | 0.35 | Fraction of passengers successfully booked (1.0 = all placed) |
| **cabin_match_score** | 0.15 | Priority-weighted fraction booked in their original cabin |
| **group_integrity_score** | 0.15 | How well groups were kept together (averaged across all groups) |
| **deadline_score** | 0.15 | Priority-weighted fraction of deadline-bearing passengers whose deadlines were met |
| **ssr_integrity_score** | 0.20 | 1.0 if no SSR violations; penalized per violation |

**Total weights sum to 1.0.**

### 7.2 Grader Formula

```
grader_score = 0.35 * coverage
             + 0.15 * cabin_match
             + 0.15 * group_integrity
             + 0.15 * deadline
             + 0.20 * ssr_integrity
             - 0.15 * hard_violation_count

grader_score = clamp(grader_score, EPS, 1 - EPS)
```

### 7.3 Component Computation Details

#### Coverage Score

```
coverage_score = n_booked / n_total
```

Simple fraction. If 20 of 25 passengers are booked: 0.80.

#### Cabin Match Score (Priority-Weighted)

For each passenger, check if they were booked in their original cabin class. Weight by priority:

```
cabin_match_score = sum(priority_weight(p) for p booked in original cabin)
                  / sum(priority_weight(p) for all p)
```

A tier-1 passenger matched in business contributes more to this score than a tier-5 passenger.

#### Group Integrity Score

Per-group scoring, then averaged:

| Scenario | Score |
|----------|-------|
| All members on same flight, same cabin | **0.7** |
| All members on same flight, different cabins | **0.5** |
| Hard group split across flights | **0.0** (+ 1 hard violation) |
| Soft group split across flights | **0.04** (no hard violation) |
| No members booked | **0.0** |
| Partially booked (some members missing) | Treated as split |

If there are no groups at all, the score defaults to **1.0**.

#### Deadline Score (Priority-Weighted)

For passengers who have a downstream deadline:

```
deadline_score = sum(priority_weight(p) for p whose deadline was met)
               / sum(priority_weight(p) for all p with deadlines)
```

If no passengers have deadlines, score defaults to **1.0**.

Time comparison: `arrival_time` is parsed as minutes since midnight and compared to `deadline`.

```python
def meets_deadline(arrival_time: str, deadline: str) -> bool:
    return parse_time(arrival_time) <= parse_time(deadline)
```

#### SSR Integrity Score

```
ssr_score = max(0.0, 1.0 - 0.25 * violation_count)
```

A violation occurs when a passenger with SSR flags is booked on a flight that doesn't support all of their SSR requirements. Each violation also counts as a **hard violation** for the penalty below.

#### Hard-Constraint Penalty

Each hard violation (SSR mismatch on a booked flight, hard group split across flights) subtracts **0.15** from the final score. This does NOT zero the score -- it preserves gradient for the remaining quality of the plan.

### 7.4 Score Clamping

The final score is clamped to **(1e-4, 1 - 1e-4)** to ensure it's always strictly between 0 and 1. This makes the score deterministic and reproducible.

---

## 8. Difficulty Tiers & Data Design

All tiers use the **same prompt**, **same tools**, **same reward function**. Difficulty comes entirely from the data.

### 8.1 Easy

**Data**: `data/easy/`

| Aspect | Value |
|--------|-------|
| Passengers | 8 |
| Groups | 0 |
| SSR flags | none |
| Deadlines | none |
| Alternative flights | 3 (FL-201, FL-202, FL-203) |
| Max steps | 30 |
| Capacity | Ample surplus (all flights support all SSRs, plenty of seats) |

**What makes it easy**: No constraints beyond cabin matching. The agent just needs to place passengers into matching cabins with enough seats. A straightforward "put everyone in their original cabin on the first available flight" strategy works.

### 8.2 Medium

**Data**: `data/medium/`

| Aspect | Value |
|--------|-------|
| Passengers | 15 |
| Groups | 2 (GRP-M01: hard, 3 members; GRP-M02: soft, 2 members) |
| SSR flags | 2 passengers (PAX-M001: UM; PAX-M009: WCHR) |
| Deadlines | 2 passengers (PAX-M006: 14:30; PAX-M012: 16:00) |
| Alternative flights | 4 (FL-201 through FL-204) |
| Max steps | 60 |
| Capacity | Moderate pressure; some flights don't support all SSRs |

**What makes it medium**: The agent must handle group bookings (using `book_group` for the hard group), check SSR compatibility before booking, and respect deadlines. Not all flights support UM or WCHR, so the agent must route SSR passengers to compatible flights.

### 8.3 Hard

**Data**: `data/hard/`

| Aspect | Value |
|--------|-------|
| Passengers | 25 |
| Groups | 4 (GRP-H01: hard, 3 members; GRP-H02: hard, 2 members; GRP-H03: soft, 3 members; GRP-H04: soft, 2 members) |
| SSR flags | 6 passengers (UM, WCHR, pet_cabin, pet_cargo across different passengers) |
| Deadlines | 5 passengers (ranging from 13:00 to 16:00) |
| Alternative flights | 5 (FL-201 through FL-205) |
| Max steps | 90 |
| Capacity | Tight (FL-204 has only 3 economy, 1 premium, 1 business; FL-205 has 2 economy, 1 premium, 1 business) |

**What makes it hard**:
- SSR scarcity: `pet_cabin` is only supported on FL-201 and FL-203; `pet_cargo` only on FL-202 and FL-204. The agent must carefully route SSR passengers.
- Group complexity: A hard group (GRP-H01) has a member with WCHR, limiting which flights the entire group can go on.
- Deadline pressure: Multiple passengers with tight deadlines (one as early as 13:00) constrain which flights are viable.
- Capacity constraints: Later flights have very few seats, forcing the agent to plan ahead rather than just filling greedily.
- Combinatorial tension: The constraints interact -- a hard group with an SSR member AND a deadline might only have one viable flight.

---

## 9. Inference Pipeline

### 9.1 Overview

`inference.py` implements a baseline agent loop that:
1. Connects to the environment via WebSocket client
2. Sends a system prompt to the LLM
3. Loops: get LLM action -> send to environment -> receive observation -> format for LLM -> repeat
4. Runs all three difficulty tiers sequentially

### 9.2 System Prompt

The system prompt (`SYSTEM_PROMPT` in `inference.py`) is **constant across all difficulty tiers**. It tells the agent:

- **Situation**: Flight cancelled, passengers need rebooking
- **Goal**: Produce a rebooking plan respecting constraints
- **Constraint priority order**: Hard constraints > coverage > cabin matching > priority tiers > soft groups > fallback
- **All 7 tools** with examples of how to call them
- **Recommended strategy**: Survey first, book constrained passengers first, use book_group for groups, finalize when done
- **Action format**: Raw JSON only, no reasoning text

### 9.3 LLM Integration

- **Model**: Configurable via `MODEL_NAME` env var (default: `Qwen/Qwen2.5-7B-Instruct`)
- **API**: OpenAI-compatible client pointed at `API_BASE_URL` (default: HuggingFace router)
- **Temperature**: 0.3 (relatively deterministic)
- **Max tokens**: 1000 per response
- **Context management**: Only the last 6 conversation history items are sent to avoid context overflow

### 9.4 Action Parsing

The agent's LLM response is parsed as JSON. The parser handles:
- Raw JSON objects
- JSON wrapped in markdown code fences
- Fallback regex extraction for `{"tool_name": ...}` patterns

If parsing fails entirely, a fallback action is used:
- If no passengers booked yet: `list_passengers`
- Otherwise: `list_alternative_flights`

### 9.5 Logging

Mandatory structured logging for evaluation:
- `[START]` with task name, environment, model
- `[STEP]` with step number, action, reward, done flag, error
- `[END]` with success flag, total steps, score, reward history

---

## 10. Deployment & Packaging

### 10.1 Docker

Multi-stage build using `ghcr.io/meta-pytorch/openenv-base`:
1. **Builder stage**: Copy project, install dependencies via `uv sync`
2. **Runtime stage**: Copy venv and project from builder, run uvicorn

```bash
# Build
docker build -t flight_rebooking .

# Run
docker run -p 8000:8000 flight_rebooking
```

Health check: hits `http://localhost:8000/health` every 30 seconds.

### 10.2 Package Structure

```toml
# pyproject.toml
[project]
name = "flight_rebooking"
version = "0.1.0"
requires-python = ">=3.10,<4.0"
```

Package maps:
- `flight_rebooking` -> repo root (`.`)
- `flight_rebooking.server` -> `server/`

### 10.3 Running Locally

```bash
# Start the server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run inference
python inference.py

# Run tests
pytest tests/ -v
```

### 10.4 OpenEnv Spec

`openenv.yaml` declares the environment as a FastAPI space on port 8000:

```yaml
spec_version: 1
name: flight_rebooking
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

### 10.5 Endpoints

All created automatically by `create_app()`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Execute one tool call |
| `/state` | GET | Current episode metadata |
| `/schema` | GET | Action/observation JSON schemas |
| `/ws` | WS | WebSocket for persistent sessions |
| `/health` | GET | Liveness check |
| `/docs` | GET | Auto-generated API docs |

---

## 11. Testing Strategy

### 11.1 Test Structure

Two test files:

- `tests/test_environment.py` -- Integration tests against the full environment (no server, no WebSocket)
- `tests/test_rewards.py` -- Unit tests for the reward computer with inline data

### 11.2 Environment Tests (`test_environment.py`)

**13 test classes** covering:

| Class | What It Tests |
|-------|--------------|
| `TestReset` | Initial observation fields, max steps, unknown task ID |
| `TestListPassengers` | Returns all passengers, summary (not full details), positive reward |
| `TestGetPassengerDetails` | Full record returned, nonexistent passenger error, already-booked penalty |
| `TestListAlternativeFlights` | All flights returned, availability decrements, SSR info, flights snapshot |
| `TestGetFlightDetails` | Full flight returned, nonexistent flight error |
| `TestBookPassenger` | Successful booking, availability decrement, same-cabin/upgrade/downgrade rewards, double-booking error, no-availability error, SSR mismatch error, deadline violation error, invalid cabin error, booked summary update |
| `TestBookGroup` | Successful group booking, atomic failure, hard group same flight, split cabin allowed, SSR check all members, nonexistent group error, partially booked group error |
| `TestFinalizePlan` | Triggers done, grader score present and in range, step-after-done raises |
| `TestEasyTask` | Reset fields, full optimal booking (grader > 0.99), no groups/SSR/deadlines, step limit termination |
| `TestMediumTask` | Reset fields, group booking, SSR respected, deadline respected, full optimal booking (grader > 0.99) |
| `TestHardTask` | Reset fields, multiple groups, SSR scarcity, capacity pressure, full optimal booking (grader > 0.99) |
| `TestInvalidTool` | Unknown tool error, empty args handled, negative reward |
| `TestEpisodeCompletion` | Auto-finalize on all booked, state.is_complete, timeout termination, grader present on timeout, terminal breakdown keys, cumulative reward accumulation |

### 11.3 Known-Optimal Assignments

Each difficulty tier has a hand-crafted **optimal assignment** that achieves grader score > 0.99. These serve as regression tests -- if the environment or reward logic changes and the optimal score drops, something is broken.

### 11.4 Reward Tests (`test_rewards.py`)

**9 test classes** with inline test data (no file I/O):

| Class | What It Tests |
|-------|--------------|
| `TestInfoCallRewards` | First list_passengers positive, repeated churn negative, get_details unbooked/booked |
| `TestBookingRewards` | Same cabin high reward, upgrade medium, downgrade small, priority weight scaling, deadline bonus, failed booking penalty, invalid tool penalty |
| `TestGraderScore` | Perfect score, zero coverage, partial coverage, SSR violation, group split, deadline missed, deterministic, clamped to EPS range |
| `TestPriorityWeights` | Tier 1 highest, tier 5 lowest, unknown defaults to 1.0 |
| `TestCabinMatchScore` | All matched (1.0), none matched (0.0), partial match with priority weighting |
| `TestGroupIntegrityScore` | Same flight same cabin, same flight diff cabin, hard group split, soft group split, no groups |
| `TestDeadlineScore` | All met, deadline missed, no deadline passengers |
| `TestSSRIntegrity` | No violations, one violation (0.75), no SSR passengers |
| `test_grader_weights_sum_to_one` | Sanity check that component weights sum to 1.0 |
