Implement the specified module: $ARGUMENTS

Before implementing, read the relevant spec documents:
- `agent_docs/migration_spec.md` — canonical design
- `agent_docs/architecture.md` — data models and patterns
- `agent_docs/testing_strategy.md` — what tests expect

Also read the existing code for the module being replaced to understand the patterns (imports, error handling, OpenEnv contract).

## Module targets (use as $ARGUMENTS):

### `data`
Create data files for all 3 tiers: `data/{easy,medium,hard}/{passengers.json,flights.json,config.json}`.
Design realistic data with documented optimal assignments. Ensure:
- Easy: no groups, no SSR, no deadlines, ample capacity
- Medium: some groups, some SSR, some deadlines, moderate pressure
- Hard: many groups, many SSR, tight deadlines, capacity scarcity
- Each tier's optimal assignment must achieve grader ≈ 1.0

### `models`
Rewrite `models.py` with FlightRebookingAction, FlightRebookingObservation, FlightRebookingState.
Inherit from openenv base types. Match field definitions in architecture.md.

### `tools`
Rewrite `server/tools.py` with all 7 tool functions.
Follow validation chains from architecture.md. Return dicts with status key.
Pure functions taking episode state + args.

### `rewards`
Rewrite `server/rewards.py` with new RewardComputer.
Implement step-level rewards, grader_score, terminal_breakdown.
Follow exact weights and formulas from migration_spec.md.

### `environment`
Rewrite `server/environment.py` with FlightRebookingEnvironment.
New EpisodeState dataclass. reset() loads JSON data. step() routes tools.
Preserve OpenEnv Environment interface.

### `tests`
Rewrite `tests/test_environment.py` and `tests/test_rewards.py`.
Cover all test classes from testing_strategy.md.
Use known-optimal assignments from data design.

### `inference`
Rewrite `inference.py` with new system prompt (constant across tiers).
Update tool calling to match new 7-tool interface.
Update state formatting for flight-based data.

### `packaging`
Update pyproject.toml, Dockerfile, openenv.yaml, __init__.py, client.py, server/__init__.py, README.md.
Rename seat_reassignment → flight_rebooking everywhere.

## Rules:
- After implementing, run the relevant tests: `pytest tests/ -v`
- If tests fail, fix before moving on
- Preserve OpenEnv contract — never break reset/step/state interface
- Use existing code patterns where possible (error handling, imports, etc.)
