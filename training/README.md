# Flight Rebooking Environment — Training Reference

Quick-reference for anyone writing rollout / training code against the env.

---

## 1. Environment Contract

### Server

FastAPI server (openenv-core). Start with `uv run server`. Default: `http://localhost:8000`.
WebSocket-based: the `FlightRebookingEnv` client in `client.py` wraps the WS protocol.

### Lifecycle

```
result = env.reset(task_id="easy"|"medium"|"hard")
obs = result.observation          # FlightRebookingObservation

for step in range(max_steps):
    result = env.step(FlightRebookingAction(tool_name=..., args={...}))
    obs    = result.observation
    reward = result.reward        # float
    done   = result.done          # bool
    if done:
        break
```

### Action (input to `step()`)

```python
class FlightRebookingAction:
    tool_name: str   # one of the 4 tools below
    args: dict       # named arguments (empty dict for info/finalize tools)
```

### Observation (returned by `reset()` and `step()`)

| Field                | Type        | Description |
|----------------------|-------------|-------------|
| `passengers_total`   | int         | Total passengers needing rebooking |
| `passengers_booked`  | int         | Accepted in current plan |
| `passengers_remaining` | int       | Not yet booked |
| `tool_result`        | dict\|None  | Result of last tool call; `None` after reset |
| `reward_reason`      | str         | Human-readable reason for last reward |
| `step_count`         | int         | Current step (0 after reset) |
| `max_steps`          | int         | Step budget (always 5 in current data) |
| `cumulative_reward`  | float       | Sum of rewards so far |
| `booked_summary`     | list[dict]  | `[{passenger_id, flight_id, cabin}, ...]` |
| `plan_submitted`     | bool        | Whether submit_plan was called |
| `done`               | bool        | Episode terminated? |
| `reward`             | float       | Reward for this step |

### State (via `env.state` property)

`FlightRebookingState` with: `episode_id`, `step_count`, `total_passengers`, `passengers_booked`, `passengers_remaining`, `cumulative_reward`, `is_complete`, `plan_submitted`.

---

## 2. Tools (4 total)

### `get_full_manifest` — args: `{}`

Returns all passengers in one call.

```json
{
  "status": "success",
  "passengers": [
    {
      "passenger_id": "PAX-E001",
      "name": "Aarav Sharma",
      "priority_tier": 1,
      "original_cabin": "business",
      "group_id": null,
      "group_integrity": null,
      "group_size": null,
      "ssr_flags": [],
      "downstream_deadline": null,
      "current_booking": {"flight_id": "FL-201", "cabin": "business"}  // only if booked
    }
  ]
}
```

**Passenger fields:**

| Field                 | Type           | Values / Notes |
|-----------------------|----------------|----------------|
| `passenger_id`        | str            | `PAX-E001`, `PAX-M003`, `PAX-H012` etc. |
| `priority_tier`       | int            | 1 (highest) to 5 (lowest) |
| `original_cabin`      | str            | `economy`, `premium_economy`, `business` |
| `group_id`            | str\|null      | e.g. `GRP-M01` |
| `group_integrity`     | str\|null      | `hard`, `soft`, or null |
| `group_size`          | int\|null      | Number of members |
| `ssr_flags`           | list[str]      | Subset of: `UM`, `WCHR`, `pet_cabin`, `pet_cargo` |
| `downstream_deadline` | str\|null      | `"HH:MM"` or null |

### `get_flight_inventory` — args: `{}`

Returns all alternative flights.

```json
{
  "status": "success",
  "flights": [
    {
      "flight_id": "FL-201",
      "departure_time": "09:00",
      "arrival_time": "12:15",
      "cabin_availability": {"economy": 10, "premium_economy": 5, "business": 4},
      "supports_ssr": ["UM", "WCHR", "pet_cabin", "pet_cargo"]
    }
  ]
}
```

### `submit_plan` — args: `{"assignments": {"PAX-001": {"flight_id": "FL-201", "cabin": "economy"}, ...}}`

Submits a complete rebooking plan. Also accepts flat format (without the `"assignments"` wrapper). **One submission per episode — no revisions.**

Validation per passenger (in order):
1. Passenger exists
2. Flight exists
3. Cabin is valid (`economy`, `premium_economy`, `business`)
4. Cabin has availability
5. SSR compatibility (flight must support all passenger SSR flags)
6. Deadline check (arrival_time <= downstream_deadline)

Returns:

```json
{
  "status": "success",
  "per_passenger": [{"passenger_id": "...", "flight_id": "...", "cabin": "...", "status": "accepted"|"rejected", "reason": "..."}],
  "group_results": [{"group_id": "...", "integrity": "hard"|"soft", "verdict": "together"|"split_across_flights"|"partially_booked"|"none_booked", "members_flights": {...}}],
  "plan_score_preview": 0.85,
  "accepted_count": 8,
  "rejected_count": 0,
  "total": 8,
  "constraint_violations": []
}
```

### `finalize_plan` — args: `{}`

Ends the episode. Sets `done=True`. If no plan was submitted, score will be low.

---

## 3. Action XML Grammar

The inference prompt instructs the LLM to produce structured chain-of-thought reasoning inside XML tags, followed by the action. The full tag protocol:

```
<observations>
Key facts about passengers and flights.
</observations>

<passenger_analysis>
Per-constrained-passenger analysis of eligible flights.
</passenger_analysis>

<strategy>
Ordered assignment strategy: SSR -> hard groups -> deadlines -> remaining.
</strategy>

<tradeoff_analysis>
Conflict resolution when constrained passengers compete for slots.
</tradeoff_analysis>

<reconsideration>
Final review of the plan before submission.
</reconsideration>

<action>
{"tool_name": "submit_plan", "args": {"assignments": {...}}}
</action>
```

**Parsing rules** (see `parse_llm_response()` in `inference_ollama.py`):

1. Extract `<action>...</action>` tags first
2. Strip markdown code fences if present inside
3. Auto-repair truncated JSON (append missing `}`)
4. Fallback: scan for `{...}` containing `"tool_name"`
5. Nested JSON search for deeply-nested structures

**Reasoning tags** (`observations`, `passenger_analysis`, `strategy`, `tradeoff_analysis`, `reconsideration`) are optional for info-gathering steps (steps 1-2) but expected for the plan submission step.

---

## 4. Grader Components & Weights

The grader produces a single score in `(EPS, 1-EPS)` where `EPS = 1e-4`.

### Formula

```
score = 0.35 * coverage
      + 0.15 * cabin_match
      + 0.15 * group_integrity
      + 0.15 * deadline
      + 0.20 * ssr_integrity
      - 0.15 * hard_violations
```

Clamped to `[EPS, 1 - EPS]`.

### Component Details

| # | Component | Weight | Computation |
|---|-----------|--------|-------------|
| 1 | **coverage_score** | 0.35 | `booked_count / total_passengers` |
| 2 | **cabin_match_score** | 0.15 | Priority-weighted fraction of passengers in their original cabin. Weights: tier 1 = 1.5, tier 2 = 1.3, tier 3 = 1.0, tier 4 = 0.8, tier 5 = 0.6 |
| 3 | **group_integrity_score** | 0.15 | Average per-group score. Same flight + same cabin = 0.7, same flight + diff cabin = 0.5, split (hard) = 0.0, split (soft) = 0.04. No groups = 1.0 |
| 4 | **deadline_score** | 0.15 | Priority-weighted fraction of deadline-bearing passengers whose deadlines are met. No deadlines = 1.0 |
| 5 | **ssr_integrity_score** | 0.20 | `max(0, 1.0 - 0.25 * ssr_violation_count)`. An SSR violation = passenger booked on flight missing required SSR support |

**Hard-violation penalty**: `-0.15` per violation. Violations come from:
- Hard group members split across flights or partially booked
- SSR-incompatible bookings

### Step-level Rewards

| Event | Reward |
|-------|--------|
| Info call (get_full_manifest / get_flight_inventory) | -0.005 |
| submit_plan (success) | `grader_preview + (-0.005)` |
| submit_plan (duplicate, rejected) | -0.10 |
| finalize_plan (with plan) | -0.005 |
| finalize_plan (no plan) | -0.10 |
| Invalid/unknown tool | -0.20 |

### Terminal Grading

When `done=True` (either via `finalize_plan` or step limit), the `tool_result` dict includes:
- `terminal_breakdown`: dict with all 5 component scores + `hard_violations` count
- `grader_score`: float, the final clamped score

---

## 5. Seed / Task-ID to Scenario Mapping

There is **no procedural seed-based generation**. Scenarios are loaded from static JSON files in `data/{task_id}/`.

| task_id | Data dir | Passengers | Flights | Constraints |
|---------|----------|------------|---------|-------------|
| `easy`  | `data/easy/` | 8 | 3 | None. No SSR, no groups, no deadlines. All flights support all SSR. Pure coverage + cabin matching. |
| `medium` | `data/medium/` | 15 | 4 | SSR flags (UM, WCHR), 1 hard group (3 members), 1 soft group (2 members), 2 deadlines. Mixed cabins. |
| `hard` | `data/hard/` | ~25 | 5 | Dense constraints: multiple SSR types, multiple hard/soft groups, tight deadlines, scarce capacity. |

### Config file (`config.json`)

```json
{"task_id": "easy", "max_steps": 5, "cancelled_flight": "FL-100", "destination": "DEL"}
```

All tiers currently have `max_steps: 5`.

### Data file schemas

**`passengers.json`**: `{"passengers": [<passenger_obj>, ...]}`
**`flights.json`**: `{"flights": [<flight_obj>, ...]}`

The `reset(task_id=...)` method loads from `data/{task_id}/` and builds internal lookup dicts.

---

## 6. Canonical Episode Flow

A perfect 4-step episode:

1. **Step 1**: `get_full_manifest()` → get all passenger data
2. **Step 2**: `get_flight_inventory()` → get all flight data
3. **Step 3**: `submit_plan(assignments={...})` → submit complete plan, receive preview score
4. **Step 4**: `finalize_plan()` → lock in, trigger terminal grading

The agent has 5 steps max, so there's 1 spare step (but no revisions are allowed on submit_plan).

---

## 7. Key Implementation Notes for Training

- The env is stateful (WebSocket). Each episode needs `reset()` then 1-4 `step()` calls.
- `submit_plan` is the only tool that takes non-empty args. The args dict is the full rebooking plan.
- The grader preview returned by `submit_plan` equals the terminal grader score (same `grader_score()` function).
- For GRPO: the reward signal to optimize is `grader_score` at terminal step (0 to 1, higher is better).
- Valid cabins: `economy`, `premium_economy`, `business`.
- SSR types: `UM` (unaccompanied minor), `WCHR` (wheelchair), `pet_cabin`, `pet_cargo`.
- Group integrity values: `hard` (must stay together or severe penalty), `soft` (prefer together).
