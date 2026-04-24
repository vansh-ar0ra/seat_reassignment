# Architecture: Flight Rebooking Environment

## 1. File Structure (Target)

```
flight_rebooking/
├── CLAUDE.md                    # Project context for Claude Code
├── agent_docs/
│   ├── migration_spec.md        # Full migration specification
│   ├── architecture.md          # This file
│   └── testing_strategy.md      # Test requirements
├── .claude/
│   └── commands/
│       ├── plan.md              # /plan command
│       ├── implement.md         # /implement command
│       └── review.md            # /review command
├── .claudeignore
├── models.py                    # Pydantic: FlightRebookingAction, Observation, State
├── client.py                    # WebSocket EnvClient
├── __init__.py                  # Package exports
├── conftest.py                  # pytest path config
├── inference.py                 # Baseline agent (OpenAI client)
├── openenv.yaml                 # OpenEnv spec
├── pyproject.toml               # Package config
├── Dockerfile                   # Multi-stage Docker build
├── README.md                    # Documentation
├── server/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py                   # FastAPI create_app()
│   ├── environment.py           # FlightRebookingEnvironment
│   ├── tools.py                 # 7 tool functions
│   ├── rewards.py               # RewardComputer + grader
│   └── requirements.txt
├── data/
│   ├── easy/
│   │   ├── passengers.json
│   │   ├── flights.json
│   │   └── config.json
│   ├── medium/
│   │   ├── passengers.json
│   │   ├── flights.json
│   │   └── config.json
│   └── hard/
│       ├── passengers.json
│       ├── flights.json
│       └── config.json
└── tests/
    ├── __init__.py
    ├── test_environment.py
    └── test_rewards.py
```

---

## 2. Pydantic Models (`models.py`)

### FlightRebookingAction

```python
class FlightRebookingAction(Action):
    tool_name: str  # list_passengers | get_passenger_details | list_alternative_flights |
                    # get_flight_details | book_passenger | book_group | finalize_plan
    args: Dict[str, Any] = {}  # Note: Any, not str — cabin_assignments is a dict
```

### FlightRebookingObservation

```python
class FlightRebookingObservation(Observation):
    # Counters
    passengers_total: int = 0
    passengers_booked: int = 0
    passengers_remaining: int = 0
    
    # Step feedback
    tool_result: Optional[dict] = None
    reward_reason: str = "Episode started"
    step_count: int = 0
    max_steps: int = 0
    cumulative_reward: float = 0.0
    
    # Summary view (lightweight — not full manifests)
    booked_summary: List[dict] = []    # [{passenger_id, flight_id, cabin}]
    flights_snapshot: Optional[List[dict]] = None  # Current availability (only after list_alternative_flights)
```

**Design choice**: The observation does NOT dump the full passenger manifest or flight list on every step. The agent must use list_passengers / list_alternative_flights to fetch that data. This keeps context manageable and mirrors real-world APIs.

### FlightRebookingState

```python
class FlightRebookingState(State):
    total_passengers: int = 0
    passengers_booked: int = 0
    passengers_remaining: int = 0
    cumulative_reward: float = 0.0
    is_complete: bool = False
```

---

## 3. Internal Episode State (`server/environment.py`)

```python
@dataclass
class EpisodeState:
    # Immutable data loaded from files
    passengers: Dict[str, dict]          # passenger_id → full record
    flights: Dict[str, dict]             # flight_id → full record (with mutable availability)
    groups: Dict[str, List[str]]         # group_id → [passenger_ids]
    config: dict                         # tier config (max_steps, etc.)
    
    # Mutable booking state
    bookings: Dict[str, dict]            # passenger_id → {flight_id, cabin}
    flight_availability: Dict[str, Dict[str, int]]  # flight_id → {cabin: count} (decremented on booking)
    
    # Tracking for reward computation
    info_calls: Dict[str, int]           # tool_name → call count
    last_booking_step: int               # step of most recent booking (for churn detection)
    passenger_details_fetched: Set[str]  # passenger_ids whose details have been fetched
    
    # Episode metadata
    task_id: str
    step_count: int
    max_steps: int
    cumulative_reward: float
    done: bool
```

---

## 4. Tool Routing in `step()`

```python
def step(self, action: FlightRebookingAction) -> FlightRebookingObservation:
    tool_name = action.tool_name
    args = action.args
    
    if tool_name == "list_passengers":
        tool_result = tool_list_passengers(ep)
    elif tool_name == "get_passenger_details":
        tool_result = tool_get_passenger_details(ep, args.get("passenger_id", ""))
    elif tool_name == "list_alternative_flights":
        tool_result = tool_list_alternative_flights(ep)
    elif tool_name == "get_flight_details":
        tool_result = tool_get_flight_details(ep, args.get("flight_id", ""))
    elif tool_name == "book_passenger":
        tool_result = tool_book_passenger(ep, args.get("passenger_id", ""),
                                          args.get("flight_id", ""),
                                          args.get("cabin", ""))
    elif tool_name == "book_group":
        tool_result = tool_book_group(ep, args.get("group_id", ""),
                                      args.get("flight_id", ""),
                                      args.get("cabin_assignments", {}))
    elif tool_name == "finalize_plan":
        tool_result = tool_finalize_plan(ep)  # triggers done
    else:
        tool_result = {"status": "error", "message": f"Unknown tool: {tool_name}"}
```

---

## 5. Booking Validation Logic (in tools.py)

### `book_passenger` validation chain:
1. Passenger exists
2. Passenger not already booked
3. Flight exists
4. Cabin is valid (economy | premium_economy | business)
5. Cabin has availability > 0
6. Flight supports ALL of passenger's SSR flags
7. If passenger has downstream_deadline → flight arrival_time <= deadline
8. If passenger is in a hard group → warn (should use book_group)

### `book_group` validation chain:
1. Group exists
2. All group members exist and none already booked
3. Flight exists
4. All cabin_assignments reference valid cabins
5. Flight has sufficient capacity for ALL members in their assigned cabins
6. Flight supports SSR flags of ALL group members
7. If any member has downstream_deadline → flight arrival_time <= deadline

---

## 6. Reward Computer Design

```python
class RewardComputer:
    def __init__(self, total_passengers: int, max_steps: int):
        ...
    
    # Step-level
    def reward_for_info_call(self, tool_name, ep_state) -> (float, str)
    def reward_for_booking(self, tool_result, passenger, flight, ep_state) -> (float, str)
    def reward_for_group_booking(self, tool_result, group_passengers, flight, ep_state) -> (float, str)
    def reward_for_failed_action(self, tool_result) -> (float, str)
    def reward_for_invalid_tool(self) -> (float, str)
    
    # Terminal
    def grader_score(self, bookings, passengers, flights, groups) -> float
    def terminal_breakdown(self, bookings, passengers, flights, groups) -> dict
```

### Priority Weight Function
```python
PRIORITY_WEIGHTS = {1: 1.5, 2: 1.3, 3: 1.0, 4: 0.8, 5: 0.6}

def priority_weight(tier: int) -> float:
    return PRIORITY_WEIGHTS.get(tier, 1.0)
```

---

## 7. Time Comparison Logic

Deadlines and flight times use HH:MM strings. Compare as minutes-since-midnight:

```python
def parse_time(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)

def meets_deadline(arrival_time: str, deadline: str) -> bool:
    return parse_time(arrival_time) <= parse_time(deadline)
```

---

## 8. Observation Building

The observation is lightweight by design. It provides:
- Counters (total, booked, remaining)
- Step metadata (step_count, max_steps, cumulative_reward)
- Last tool_result (detailed feedback)
- Booking summary (what's been booked so far — needed for agent decision-making)
- flights_snapshot: only populated after list_alternative_flights is called

This prevents flooding the agent's context with data it hasn't asked for.

---

## 9. Episode Termination

Episode ends when ANY of:
1. `finalize_plan()` is called
2. Step limit reached (`step_count >= max_steps`)
3. All passengers booked (auto-finalize)

On termination, the grader score is computed and attached to the final tool_result.
