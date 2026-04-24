# Migration Specification: Seat Reassignment → Flight Rebooking

This is the canonical design document. All implementation MUST conform to this spec.

---

## 1. Problem Statement

A scheduled flight has been cancelled. Passengers must be rebooked onto alternative flights to the same destination within a defined time window. The agent operates at the **inventory level** — placing passengers into available cabin buckets on alternative flights, NOT assigning specific seats.

The agent must produce a rebooking plan that:
- Gets passengers to their destination
- Respects hard constraints (SSR compatibility, hard group integrity, deadlines)
- Honors priority tiers
- Keeps groups whole
- Accommodates special service needs
- Protects downstream connections
- Makes sound ancillary service decisions (hotels, meals, lounge, transport)

---

## 2. Data Structures

### 2.1 Passenger Record

```python
{
    "passenger_id": "PAX-001",
    "name": "Aarav Sharma",
    "priority_tier": 1,           # ordinal 1-5, lower = higher priority
    "original_cabin": "business",  # economy | premium_economy | business
    "group_id": "GRP-001" | null,  # nullable
    "group_integrity": "hard" | "soft" | null,  # only set if group_id present
    "group_size": 3 | null,
    "ssr_flags": ["UM"] | [],      # subset of: UM, WCHR, pet_cabin, pet_cargo
    "downstream_deadline": "14:30" | null  # HH:MM, nullable
}
```

### 2.2 Alternative Flight Record

```python
{
    "flight_id": "FL-201",
    "departure_time": "10:30",
    "arrival_time": "13:45",
    "cabin_availability": {
        "economy": 45,
        "premium_economy": 12,
        "business": 4
    },
    "supports_ssr": ["WCHR", "pet_cargo"]  # which SSRs this flight can accommodate
}
```

### 2.3 Ancillary Service Menu (future — stub for now)

```python
{
    "hotel": {"cost_to_airline": 150, "goodwill_base": 50},
    "meal_voucher": {"cost_to_airline": 25, "goodwill_base": 15},
    "lounge_access": {"cost_to_airline": 40, "goodwill_base": 30},
    "ground_transport": {"cost_to_airline": 60, "goodwill_base": 20}
}
```

---

## 3. Tool Specifications

### 3.1 Information-Gathering Tools

#### `list_passengers()`
- **Args**: none
- **Returns**: Summary list of all passengers: passenger_id, priority_tier, group_id, constraint hints (has_ssr, has_deadline)
- **Use**: Called once early to plan ordering. Lightweight.

#### `get_passenger_details(passenger_id)`
- **Args**: `passenger_id: str`
- **Returns**: Full record — original_cabin, exact SSR flags, group membership + integrity_type, downstream_deadline, priority_tier
- **Use**: Called per-passenger when ready to work on them

#### `list_alternative_flights()`
- **Args**: none
- **Returns**: Full pool of alternative flights with per-cabin seat counts, departure/arrival times, SSR support
- **Behavior**: Seat counts update as agent books — repeat calls are valid
- **Use**: Survey capacity, called after bookings to refresh availability

#### `get_flight_details(flight_id)`
- **Args**: `flight_id: str`
- **Returns**: Everything about a specific flight including current per-cabin availability
- **Use**: Precise check before committing

### 3.2 Commitment Tools

#### `book_passenger(passenger_id, flight_id, cabin)`
- **Args**: `passenger_id: str`, `flight_id: str`, `cabin: str` (economy|premium_economy|business)
- **Returns**: success/failure with reason. On success, decrements flight's cabin count.
- **Validation**:
  - Passenger must exist and not already booked
  - Flight must exist
  - Cabin must have availability > 0
  - Flight must support passenger's SSR flags (if any)
  - If passenger has a downstream_deadline, arrival_time must be <= deadline
  - If passenger is in a hard group, warn (prefer book_group for hard groups)
- **On failure**: explain why (SSR mismatch, no seats, deadline violation, etc.)

#### `book_group(group_id, flight_id, cabin_assignments)`
- **Args**: `group_id: str`, `flight_id: str`, `cabin_assignments: dict[str, str]` mapping passenger_id → cabin
- **Returns**: success/failure. Atomic — all or none booked.
- **Validation**:
  - All passengers in group must exist and not be booked
  - Flight must have capacity for all members in their assigned cabins
  - Flight must support SSR flags of ALL group members
  - For hard groups: all must be on the same flight (enforced by design — single flight_id)
  - cabin_assignments allows split-cabin on same flight (fallback)
- **On failure**: nothing changes, explain why

### 3.3 Finalization

#### `finalize_plan()`
- **Args**: none
- **Returns**: triggers end-of-episode grading. Unbooked passengers = coverage failures.
- **Behavior**: episode ends, grader computes final score

---

## 4. Reward Function Design

### 4.1 Step-Level Rewards (per action)

**Information tools:**
| Call | Reward |
|------|--------|
| list_passengers() first time | +0.02 |
| list_passengers() 5th+ time with no intervening bookings | -0.01 |
| get_passenger_details(pid) for unbooked passenger | +0.02 |
| get_passenger_details(pid) for already-booked passenger | -0.01 |
| list_alternative_flights() | +0.01 (always useful due to inventory changes) |
| get_flight_details(fid) | +0.01 |

**Booking outcomes:**

| Outcome | Step Reward |
|---------|-------------|
| Hard-constraint violation (SSR mismatch, hard group split across flights, deadline missed when alternative existed) | -0.30 |
| Successful booking, same cabin, whole group together | +0.15 × priority_weight |
| Successful booking, upgrade from original | +0.10 × priority_weight |
| Successful booking, split across cabins same flight (fallback) | +0.05 × priority_weight |
| Successful booking, cabin downgrade | +0.02 × priority_weight |
| Successful booking + downstream deadline met | additional +0.05 × priority_weight |
| Successful booking, deadline missed but no alternative existed | neutral (0.0) |
| Failed booking (rejected by environment) | -0.02 |

**Priority weights:**
- Tier 1 → 1.5
- Tier 2 → 1.3  
- Tier 3 → 1.0
- Tier 4 → 0.8
- Tier 5 → 0.6

**finalize_plan():** triggers settlement, no step reward itself.

### 4.2 End-of-Episode Settlement (Grader Components)

| Component | Weight | Description |
|-----------|--------|-------------|
| coverage_score | 0.35 | Fraction of passengers successfully booked (1.0 = all placed) |
| cabin_match_score | 0.15 | Fraction placed in original cabin (priority-weighted) |
| group_integrity_score | 0.15 | Per-group: 1.0 same-flight-same-cabin, 0.7 same-flight-diff-cabin, 0.0 split-flights (hard), 0.4 split-flights (soft). Averaged. |
| deadline_score | 0.15 | Fraction of deadline-bearing passengers whose deadlines met (priority-weighted) |
| ssr_integrity_score | 0.20 | 1.0 if no SSR violations, penalized sharply per violation |

**Grader formula:**
```
grader_score = (
    0.35 × coverage_score
  + 0.15 × cabin_match_score
  + 0.15 × group_integrity_score
  + 0.15 × deadline_score
  + 0.20 × ssr_integrity_score
)
grader_score = max(EPS, min(1 - EPS, grader_score))
```
where EPS = 1e-4.

### 4.3 Hard-Constraint Violation Override

Each hard violation (SSR on wrong flight, hard-group split across flights) subtracts 0.15 from final grader_score. Does NOT zero it out — preserves gradient for remaining passengers.

---

## 5. Agent Prompt Structure

The system prompt is **constant across all three tiers**. It communicates:

1. **Situation**: Flight cancelled, passengers need rebooking
2. **Objective**: Produce rebooking plan respecting constraints and priorities
3. **Tools**: Brief description of all 7 tools, emphasizing list_passengers first, finalize_plan last
4. **Priority ordering**:
   - Hard constraints must not be violated (SSR, hard groups, deadlines where feasible)
   - Every passenger should be rebooked
   - Match original cabin where possible
   - Higher-tier passengers get better outcomes on trade-offs
   - Keep soft groups together where possible
   - Split across cabins on same flight before splitting across flights
5. **Recommended workflow**: survey → order by constraint scarcity → evaluate → commit → iterate

The prompt does NOT include specific numbers (passenger counts, flight counts, reward weights).

---

## 6. Difficulty Tiers (Data Design)

All tiers use the SAME prompt, SAME tools, SAME reward function. Difficulty is entirely in the data.

### Easy
- ~8-10 passengers
- No groups
- No SSR flags
- No deadlines
- Ample flight capacity (clear surplus)
- 2-3 alternative flights
- Max steps: ~30

### Medium
- ~15-20 passengers
- 1-2 groups (1 hard, 1 soft)
- 2-3 passengers with SSR flags
- 2-3 passengers with deadlines
- Moderate capacity pressure
- 3-4 alternative flights
- Max steps: ~60

### Hard
- ~25-30 passengers
- 3+ groups (mix of hard/soft)
- 5+ passengers with SSR flags
- 5+ passengers with deadlines
- Tight capacity (some cabins at limit)
- 4-5 alternative flights (not all support all SSRs)
- Max steps: ~90

---

## 7. Data File Format

Each tier directory (`data/{easy,medium,hard}/`) contains:

### `passengers.json`
```json
{
    "passengers": [
        {
            "passenger_id": "PAX-001",
            "name": "...",
            "priority_tier": 1,
            "original_cabin": "business",
            "group_id": null,
            "group_integrity": null,
            "group_size": null,
            "ssr_flags": [],
            "downstream_deadline": null
        }
    ]
}
```

### `flights.json`
```json
{
    "flights": [
        {
            "flight_id": "FL-201",
            "departure_time": "10:30",
            "arrival_time": "13:45",
            "cabin_availability": {
                "economy": 45,
                "premium_economy": 12,
                "business": 4
            },
            "supports_ssr": ["WCHR", "pet_cargo"]
        }
    ]
}
```

### `config.json` (tier metadata)
```json
{
    "task_id": "easy",
    "max_steps": 30,
    "cancelled_flight": "FL-100",
    "destination": "DEL"
}
```

---

## 8. Implementation Order

Execute in this exact sequence:

1. **Data files** — Create `data/{easy,medium,hard}/` with passengers.json, flights.json, config.json
2. **models.py** — `FlightRebookingAction`, `FlightRebookingObservation`, `FlightRebookingState`
3. **server/tools.py** — All 7 tool functions
4. **server/rewards.py** — `RewardComputer` with step rewards + grader
5. **server/environment.py** — `FlightRebookingEnvironment` with EpisodeState, reset, step
6. **tests/** — Full test suite for all 3 tiers
7. **inference.py** — New system prompt + agent loop
8. **Packaging** — pyproject.toml, Dockerfile, openenv.yaml, __init__.py, client.py, README.md
