Plan the implementation for the current task.

Before writing any code, do the following:

1. Read `agent_docs/migration_spec.md` for the full design specification
2. Read `agent_docs/architecture.md` for data models and internal state design
3. Read `agent_docs/testing_strategy.md` for test requirements
4. Examine the existing code that will be replaced — understand the patterns used in the current `server/environment.py`, `server/tools.py`, `server/rewards.py`, and `models.py`
5. Check the OpenEnv interface contract by reading the current `server/app.py` and how `create_app()` is called

Then produce a detailed implementation plan that covers:

- Which files to create/modify/delete, in what order
- For each file: what classes/functions to implement, with signatures
- Data file design: exact JSON structures for easy/medium/hard tiers
- Known-optimal assignments for each tier (for test verification)
- Dependencies between files (what must exist before what)
- Risk areas and edge cases to watch for

Format the plan as a numbered checklist. Do NOT write code yet — just plan.

The implementation order MUST be:
1. Data files (passengers.json, flights.json, config.json for all 3 tiers)
2. models.py (FlightRebookingAction, FlightRebookingObservation, FlightRebookingState)
3. server/tools.py (all 7 tool functions)
4. server/rewards.py (RewardComputer with step rewards + grader)
5. server/environment.py (FlightRebookingEnvironment)
6. tests/ (test_environment.py + test_rewards.py)
7. inference.py (new system prompt + agent loop)
8. Packaging (pyproject.toml, Dockerfile, openenv.yaml, __init__.py, client.py, README.md)
