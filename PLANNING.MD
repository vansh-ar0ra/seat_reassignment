---
  Phase 1: Data Files

  1.1 data/easy/config.json

  {"task_id": "easy", "max_steps": 30, "cancelled_flight": "FL-100", "destination": "DEL"}

  1.2 data/easy/passengers.json

  - 8 passengers: PAX-E001 through PAX-E008
  - priority_tier: mix of 1–3 (keep it simple)
  - original_cabin: 3 business, 5 economy
  - No groups (all group_id: null)
  - No SSR flags (all ssr_flags: [])
  - No deadlines (all downstream_deadline: null)

  1.3 data/easy/flights.json

  - 3 alternative flights: FL-201, FL-202, FL-203
  - Ample capacity — each flight has surplus seats in every cabin
  - All flights supports_ssr: ["UM", "WCHR", "pet_cabin", "pet_cargo"] (doesn't matter since no SSRs)
  - Arrival times all reasonable (no deadline pressure)

  Optimal easy assignment (all same-cabin, all on FL-201 since ample capacity):
  PAX-E001 → (FL-201, business)    PAX-E005 → (FL-201, economy)
  PAX-E002 → (FL-201, business)    PAX-E006 → (FL-202, economy)
  PAX-E003 → (FL-201, business)    PAX-E007 → (FL-202, economy)
  PAX-E004 → (FL-202, economy)     PAX-E008 → (FL-203, economy)
  All 8 booked, same cabin, no groups/SSR/deadlines → grader ≈ 1.0.

  1.4 data/medium/config.json

  {"task_id": "medium", "max_steps": 60, "cancelled_flight": "FL-100", "destination": "DEL"}

  1.5 data/medium/passengers.json

  - 15 passengers: PAX-M001 through PAX-M015
  - priority_tier: 1–5 distribution
  - original_cabin: 4 business, 3 premium_economy, 8 economy
  - 2 groups:
    - GRP-M01 (hard, size 3): PAX-M003, PAX-M004, PAX-M005 — all economy
    - GRP-M02 (soft, size 2): PAX-M010, PAX-M011 — both business
  - 2 SSR passengers: PAX-M001 has ["UM"], PAX-M009 has ["WCHR"]
  - 2 deadline passengers: PAX-M006 has "14:30", PAX-M012 has "16:00"

  1.6 data/medium/flights.json

  - 4 alternative flights: FL-201, FL-202, FL-203, FL-204
  - Moderate capacity — enough for all but not wildly excess
  - FL-201: supports ["UM", "WCHR"], departs 09:00, arrives 12:15
  - FL-202: supports ["WCHR", "pet_cargo"], departs 11:00, arrives 14:15
  - FL-203: supports ["UM"], departs 13:00, arrives 16:15
  - FL-204: supports ["WCHR"], departs 15:00, arrives 18:15
  - Key: FL-201 and FL-202 arrive before 14:30 deadline. FL-201/FL-202/FL-203 arrive before 16:00 deadline.

  Optimal medium assignment: All 15 booked, same cabin, hard group together on same flight/cabin, SSR passengers on compatible flights, deadlines met.
  (Exact mapping documented in test file.)

  1.7 data/hard/config.json

  {"task_id": "hard", "max_steps": 90, "cancelled_flight": "FL-100", "destination": "DEL"}

  1.8 data/hard/passengers.json

  - 25 passengers: PAX-H001 through PAX-H025
  - priority_tier: full 1–5 spread
  - original_cabin: 6 business, 4 premium_economy, 15 economy
  - 4 groups:
    - GRP-H01 (hard, size 3): economy passengers
    - GRP-H02 (hard, size 2): business passengers
    - GRP-H03 (soft, size 3): economy passengers
    - GRP-H04 (soft, size 2): premium_economy passengers
  - 6 SSR passengers: mix of UM, WCHR, pet_cabin, pet_cargo
  - 5 deadline passengers: varying deadlines from 13:00 to 17:00

  1.9 data/hard/flights.json

  - 5 alternative flights: FL-201 through FL-205
  - Tight capacity — some cabins at exactly the minimum needed
  - Not all flights support all SSRs (key difficulty driver)
  - Staggered departure/arrival times creating deadline tension

  Optimal hard assignment: All 25 booked, documented mapping achieving grader ≈ 1.0. Requires careful SSR/deadline/group routing.

  1.10 Delete old data files

  - Remove from all tiers: ac1_config.json, ac2_config.json, seats_ac1.csv, seats_ac2.csv, assignments.csv, passengers.csv
  - Remove: data/medium/passengers.json (old format), data/medium/generate_data.py, data/hard/generate_data.py

  ---
  Phase 2: models.py — Rewrite

  2.1 FlightRebookingAction(Action)

  tool_name: str  # list_passengers | get_passenger_details | list_alternative_flights |
                  # get_flight_details | book_passenger | book_group | finalize_plan
  args: Dict[str, Any] = {}  # Any, not str — cabin_assignments is a dict
  - Key change from old: args is Dict[str, Any] not Dict[str, str] because book_group.cabin_assignments is a nested dict.

  2.2 FlightRebookingObservation(Observation)

  passengers_total: int = 0
  passengers_booked: int = 0
  passengers_remaining: int = 0
  tool_result: Optional[dict] = None
  reward_reason: str = "Episode started"
  step_count: int = 0
  max_steps: int = 0
  cumulative_reward: float = 0.0
  booked_summary: List[dict] = []           # [{passenger_id, flight_id, cabin}]
  flights_snapshot: Optional[List[dict]] = None  # only after list_alternative_flights
  - Key change: Replaces ac1/ac2 layout fields with booking-centric fields. Observation is lightweight — agent must call tools to get data.

  2.3 FlightRebookingState(State)

  total_passengers: int = 0
  passengers_booked: int = 0
  passengers_remaining: int = 0
  cumulative_reward: float = 0.0
  is_complete: bool = False

  ---
  Phase 3: server/tools.py — Rewrite (7 functions)

  All functions take EpisodeState as first arg, return dict with "status" key.

  3.1 tool_list_passengers(ep) → dict

  - Returns list of {passenger_id, priority_tier, group_id, has_ssr, has_deadline} for all passengers
  - Lightweight — no full details

  3.2 tool_get_passenger_details(ep, passenger_id) → dict

  - Returns full record: original_cabin, ssr_flags, group_id, group_integrity, group_size, downstream_deadline, priority_tier, name
  - Adds passenger_id to ep.passenger_details_fetched
  - Error if passenger doesn't exist

  3.3 tool_list_alternative_flights(ep) → dict

  - Returns all flights with current cabin_availability, departure/arrival times, supports_ssr
  - Availability reflects current state (decremented by bookings)

  3.4 tool_get_flight_details(ep, flight_id) → dict

  - Returns single flight's full details with current availability
  - Error if flight doesn't exist

  3.5 tool_book_passenger(ep, passenger_id, flight_id, cabin) → dict

  - Validation chain (in order):
    a. Passenger exists
    b. Passenger not already booked
    c. Flight exists
    d. Cabin is valid enum
    e. Cabin has availability > 0
    f. Flight supports ALL of passenger's SSR flags
    g. If downstream_deadline → arrival_time <= deadline
    h. If hard group → warn (should use book_group)
  - On success: decrements ep.flight_availability[flight_id][cabin], adds to ep.bookings
  - Returns {status, passenger_id, flight_id, cabin, cabin_match, deadline_met, warnings[]}

  3.6 tool_book_group(ep, group_id, flight_id, cabin_assignments) → dict

  - Atomic — all or none
  - Validation chain:
    a. Group exists
    b. All members exist and none already booked
    c. Flight exists
    d. All cabins valid
    e. Sufficient capacity for all members
    f. Flight supports SSR flags of ALL members
    g. Deadline check for all members
  - On success: decrements availability for all, adds all to bookings
  - Returns {status, group_id, flight_id, booked: [{passenger_id, cabin}]}

  3.7 tool_finalize_plan(ep) → dict

  - Sets ep.done = True
  - Triggers grader computation
  - Returns {status: "success", grader_score, breakdown}

  3.8 Helper: parse_time(t: str) → int

  - Converts "HH:MM" to minutes since midnight

  3.9 Helper: meets_deadline(arrival_time, deadline) → bool

  - parse_time(arrival_time) <= parse_time(deadline)

  ---
  Phase 4: server/rewards.py — Rewrite

  4.1 Constants

  PRIORITY_WEIGHTS = {1: 1.5, 2: 1.3, 3: 1.0, 4: 0.8, 5: 0.6}
  EPS = 1e-4
  Plus all step reward constants from spec (§4.1).

  4.2 RewardComputer.__init__(self, total_passengers, max_steps)

  4.3 Step-level methods

  - reward_for_info_call(self, tool_name, ep_state) → (float, str) — handles list_passengers churn detection, get_passenger_details for booked
  passenger, etc.
  - reward_for_booking(self, tool_result, passenger, ep_state) → (float, str) — same-cabin/upgrade/downgrade/split-cabin logic, priority weighting,
  deadline bonus
  - reward_for_group_booking(self, tool_result, group_passengers, ep_state) → (float, str) — sums individual booking rewards
  - reward_for_failed_action(self, tool_result) → (float, str) — -0.02
  - reward_for_hard_violation(self) → (float, str) — -0.30
  - reward_for_invalid_tool(self) → (float, str) — negative reward

  4.4 Grader methods

  - grader_score(self, bookings, passengers, flights, groups) → float
    - coverage_score: fraction booked (0.35 weight)
    - cabin_match_score: priority-weighted fraction in original cabin (0.15)
    - group_integrity_score: per-group scoring (1.0/0.7/0.0/0.4), averaged (0.15)
    - deadline_score: priority-weighted fraction meeting deadlines (0.15)
    - ssr_integrity_score: 1.0 minus per-violation penalty (0.20)
    - Hard-constraint violation override: each subtracts 0.15 from final score
    - Clamped to (EPS, 1-EPS)
  - terminal_breakdown(self, ...) → dict — returns all component scores

  4.5 Helper: priority_weight(tier) → float

  ---
  Phase 5: server/environment.py — Rewrite

  5.1 EpisodeState dataclass

  @dataclass
  class EpisodeState:
      passengers: Dict[str, dict]
      flights: Dict[str, dict]
      groups: Dict[str, List[str]]
      config: dict
      bookings: Dict[str, dict]               # passenger_id → {flight_id, cabin}
      flight_availability: Dict[str, Dict[str, int]]
      info_calls: Dict[str, int]              # tool_name → call count
      last_booking_step: int
      passenger_details_fetched: Set[str]
      task_id: str
      step_count: int
      max_steps: int
      cumulative_reward: float
      done: bool

  5.2 FlightRebookingEnvironment(Environment)

  reset(self, seed=None, task_id="medium") → FlightRebookingObservation
  - Load config.json, passengers.json, flights.json from data/{task_id}/
  - Build passengers dict, flights dict, groups dict
  - Initialize flight_availability as deep copy of cabin_availability
  - Initialize empty bookings, counters, tracking sets
  - Create RewardComputer
  - Return initial observation

  step(self, action: FlightRebookingAction) → FlightRebookingObservation
  - Increment step counter
  - Route to tool function based on action.tool_name
  - Compute step reward via RewardComputer
  - Check termination: finalize called, step limit, or all booked
  - If terminal: compute grader, attach to tool_result
  - Build and return observation

  state property → FlightRebookingState

  _build_observation(...) → FlightRebookingObservation
  - Populate counters, booked_summary, flights_snapshot (if list_alternative_flights was called)

  5.3 Termination conditions (any of):

  1. finalize_plan() called
  2. step_count >= max_steps
  3. All passengers booked (auto-finalize)

  ---
  Phase 6: Tests

  6.1 tests/test_rewards.py — Complete rewrite

  Test classes per the testing strategy doc:
  - TestInfoCallRewards — 5 tests
  - TestBookingRewards — 8 tests
  - TestGraderScore — 8 tests
  - TestPriorityWeights — 3 tests
  - TestCabinMatchScore — 3 tests
  - TestGroupIntegrityScore — 5 tests

  All tests construct data inline (no file I/O). Import and test against exported constants.

  6.2 tests/test_environment.py — Complete rewrite

  Define optimal assignments at module level:
  OPTIMAL_EASY = {"PAX-E001": ("FL-201", "business"), ...}
  OPTIMAL_MEDIUM = {"PAX-M001": ("FL-201", "business"), ...}
  OPTIMAL_HARD = {"PAX-H001": ("FL-201", "business"), ...}

  Test classes (12 classes, ~50+ tests):
  - TestReset — 4 tests
  - TestListPassengers — 3 tests
  - TestGetPassengerDetails — 3 tests
  - TestListAlternativeFlights — 3 tests
  - TestGetFlightDetails — 2 tests
  - TestBookPassenger — 12 tests (includes SSR, deadline, cabin validation)
  - TestBookGroup — 7 tests (atomic failure, SSR all-members, partial-booked)
  - TestFinalizePlan — 4 tests
  - TestEasyTask — 4 tests (full optimal → grader ≈ 1.0)
  - TestMediumTask — 5 tests (groups, SSR, deadlines, full optimal)
  - TestHardTask — 5 tests (multiple groups, SSR scarcity, capacity pressure)
  - TestInvalidTool — 2 tests

  Helpers:
  def make_env(task_id="medium") → FlightRebookingEnvironment
  def call_tool(env, tool_name, **args) → FlightRebookingObservation

  ---
  Phase 7: inference.py — Rewrite

  7.1 System prompt (constant for all tiers)

  - Describes: cancelled flight, rebooking objective
  - Lists all 7 tools with brief descriptions
  - Priority ordering: hard constraints > coverage > cabin match > priority tiers > soft groups
  - Recommended workflow: survey → order → commit → iterate → finalize
  - Does NOT mention specific numbers or reward weights

  7.2 Tool definitions for OpenAI function calling

  - Define 7 tools as OpenAI-format function schemas
  - list_passengers: no params
  - get_passenger_details: passenger_id
  - list_alternative_flights: no params
  - get_flight_details: flight_id
  - book_passenger: passenger_id, flight_id, cabin
  - book_group: group_id, flight_id, cabin_assignments
  - finalize_plan: no params

  7.3 Agent loop

  - Same structure as current: run_task() async loop
  - Update TASKS list: max_steps from config (30/60/90)
  - Update format_state() for new observation fields
  - Update fallback_action() to call list_passengers
  - Update BENCHMARK to "flight_rebooking"
  - Update imports to use FlightRebookingEnv, FlightRebookingAction

  ---
  Phase 8: Packaging

  8.1 openenv.yaml

  spec_version: 1
  name: flight_rebooking
  type: space
  runtime: fastapi
  app: server.app:app
  port: 8000
  version: "1.0.0"
  description: An environment for managing airline flight rebooking after cancellation.

  8.2 pyproject.toml

  - Rename seat_reassignment → flight_rebooking everywhere
  - Update description, package names, entry point
  - Remove pandas dependency (no longer needed — JSON-based, not CSV/DataFrame)
  - Keep: openenv-core, pydantic, openai, pytest

  8.3 Dockerfile

  - Update ENV_NAME=flight_rebooking
  - Update CMD: flight_rebooking.server.app:app

  8.4 server/app.py

  - Import FlightRebookingAction, FlightRebookingObservation, FlightRebookingEnvironment
  - env_name="flight_rebooking"

  8.5 __init__.py

  - Export FlightRebookingAction, FlightRebookingObservation, FlightRebookingState, FlightRebookingEnv

  8.6 client.py

  - Rename class to FlightRebookingEnv
  - Update model imports
  - Update _step_payload, _parse_result, _parse_state for new models

  8.7 server/__init__.py

  - Update docstring

  8.8 conftest.py

  - No changes needed (path setup is generic)

  8.9 README.md — Complete rewrite for new problem

  ---
  Dependency Order (Critical Path)

  data files ──→ models.py ──→ server/tools.py ──→ server/rewards.py ──→ server/environment.py
                                                                                │
                                                                                ├──→ tests/
                                                                                └──→ inference.py ──→ packaging

  - tools.py depends on models.py (EpisodeState shape) but NOT on rewards
  - rewards.py is stateless — only depends on data shapes from models
  - environment.py imports tools + rewards + models — must come after all three
  - tests import environment + rewards + models — must come after environment
  - inference.py imports client + models — can be done after models but tested after environment

  ---
  Risk Areas & Edge Cases

  1. args: Dict[str, Any] vs Dict[str, str] — The old models used Dict[str, str]. The new book_group tool needs cabin_assignments: dict[str, str] nested
   inside args. Must be Any.
  2. Atomic group booking — book_group must validate ALL members before committing ANY. If capacity check passes but SSR fails for member 3, members 1-2
   must not be decremented.
  3. Hard group via book_passenger — Spec says "warn" but don't block. A hard-group member CAN be booked individually (triggers -0.30 penalty if they
  end up split across flights, but the booking itself succeeds). The penalty is at grading time, not booking time.
  4. SSR subset check — flight.supports_ssr must be a superset of passenger.ssr_flags. A passenger with ["UM", "WCHR"] needs a flight supporting both.
  5. Deadline comparison — String "HH:MM" comparison. Must use parse_time() converting to minutes. Edge case: arrival exactly at deadline should pass
  (<=).
  6. flights_snapshot in observation — Only populated after list_alternative_flights is called. Before that, it's None. Must persist across steps (not
  reset each step).
  7. Grader EPS — Spec says EPS = 1e-4. Old code used EPS = 0.01. Must use 1e-4 per spec.
  8. Group integrity scoring — Groups with NO booked members score 0.0 (they're coverage failures). Groups with ALL members booked on same flight/same
  cabin score 1.0. Must handle partial booking of groups.
  9. No pandas — Old code used DataFrames heavily. New code should use plain dicts/lists since data is JSON-based. This is cleaner and removes the
  pandas dependency.
  10. info_calls churn detection — list_passengers called 5+ times with no intervening bookings gets penalty. Must track last_booking_step to detect "no
   intervening bookings."
  11. Auto-finalize on all booked — If the last book_passenger/book_group call books the final passenger(s), the episode auto-terminates. Grader must be
   computed on that step.
  12. cabin_assignments in book_group — Allows split-cabin on same flight (e.g., some members in economy, some in premium_economy). This is valid for
  both hard and soft groups per spec.

  ---
  Summary Checklist

  ┌─────┬───────────────────────────────────┬──────────────────────────────────────────────┬────────────┐
  │  #  │               Task                │                    Files                     │ Depends On │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 1   │ Create easy data files            │ data/easy/{config,passengers,flights}.json   │ —          │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 2   │ Create medium data files          │ data/medium/{config,passengers,flights}.json │ —          │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 3   │ Create hard data files            │ data/hard/{config,passengers,flights}.json   │ —          │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 4   │ Delete old data files             │ All old CSV/JSON in data/                    │ —          │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 5   │ Rewrite models.py                 │ models.py                                    │ —          │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 6   │ Rewrite server/tools.py           │ server/tools.py                              │ #5         │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 7   │ Rewrite server/rewards.py         │ server/rewards.py                            │ #5         │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 8   │ Rewrite server/environment.py     │ server/environment.py                        │ #5, #6, #7 │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 9   │ Rewrite tests/test_rewards.py     │ tests/test_rewards.py                        │ #7         │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 10  │ Rewrite tests/test_environment.py │ tests/test_environment.py                    │ #8         │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 11  │ Run pytest — all green            │ —                                            │ #9, #10    │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 12  │ Rewrite inference.py              │ inference.py                                 │ #5         │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 13  │ Update server/app.py              │ server/app.py                                │ #5, #8     │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 14  │ Update client.py                  │ client.py                                    │ #5         │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 15  │ Update init.py                    │ __init__.py                                  │ #5         │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 16  │ Update server/init.py             │ server/__init__.py                           │ —          │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 17  │ Update openenv.yaml               │ openenv.yaml                                 │ —          │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 18  │ Update pyproject.toml             │ pyproject.toml                               │ —          │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 19  │ Update Dockerfile                 │ Dockerfile                                   │ —          │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 20  │ Rewrite README.md                 │ README.md                                    │ —          │
  ├─────┼───────────────────────────────────┼──────────────────────────────────────────────┼────────────┤
  │ 21  │ Final pytest + server startup     │ —                                            │ All        │
  └─────┴───────────────────────────────────┴──────────────────────────────────────────────┴────────────┘

  Ready to implement on your go. Use /implement <module> to start any phase, or tell me to begin with Phase 1 (data files).