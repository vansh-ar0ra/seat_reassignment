Review the current implementation against the spec.

## Review Checklist

### 1. Spec Compliance
Read `agent_docs/migration_spec.md` and verify:
- [ ] All 7 tools implemented with correct signatures
- [ ] Reward weights match spec (0.35/0.15/0.15/0.15/0.20)
- [ ] Priority weights correct (Tier1→1.5, Tier3→1.0, Tier5→0.6)
- [ ] Grader clamped to (EPS, 1-EPS) where EPS=1e-4
- [ ] Step rewards match the outcome table
- [ ] Hard-constraint violations subtract 0.15 per violation
- [ ] System prompt is identical across all 3 tiers
- [ ] book_group is atomic (all-or-nothing)

### 2. OpenEnv Contract
- [ ] `FlightRebookingEnvironment` inherits from `Environment`
- [ ] `reset(seed, task_id)` returns `FlightRebookingObservation`
- [ ] `step(action)` takes `FlightRebookingAction`, returns `FlightRebookingObservation`
- [ ] `state` property returns `FlightRebookingState`
- [ ] Models inherit from Action, Observation, State base types
- [ ] `create_app()` call in server/app.py uses new class names
- [ ] openenv.yaml name matches package name

### 3. Data Integrity
- [ ] All 3 tiers have passengers.json, flights.json, config.json
- [ ] Easy: no groups, no SSR, no deadlines
- [ ] Medium: has groups + SSR + deadlines
- [ ] Hard: tight capacity, many constraints
- [ ] Each tier has a documented optimal assignment achieving grader ≈ 1.0
- [ ] Flight SSR support is consistent (flights declare what they support)
- [ ] Cabin availability makes optimal solution feasible

### 4. Tool Validation
For each booking tool, verify the validation chain:
- [ ] book_passenger: exists, not booked, flight exists, cabin valid, availability > 0, SSR check, deadline check
- [ ] book_group: group exists, all unbooked, flight exists, cabins valid, capacity for all, SSR for all, deadline for all
- [ ] Info tools: handle nonexistent IDs gracefully
- [ ] finalize_plan: triggers grading, sets done=True

### 5. Test Coverage
Run `pytest tests/ -v` and verify:
- [ ] All tests pass
- [ ] Each tier has a full-episode optimal test
- [ ] Error paths tested (invalid IDs, double booking, SSR mismatch, etc.)
- [ ] Reward values tested against constants
- [ ] Grader score tested for determinism and range

### 6. Packaging
- [ ] pyproject.toml has correct package name and entry points
- [ ] Dockerfile CMD references correct module path
- [ ] All imports resolve (try `python -c "from models import FlightRebookingAction"`)
- [ ] `uv run server` starts without error

### 7. Code Quality
- [ ] No hardcoded step limits (comes from data config)
- [ ] No seat-level references remaining (old code artifacts)
- [ ] Consistent error dict format: {"status": "error"|"success", "message": "..."}
- [ ] Type hints on all public functions
- [ ] No unused imports

## Output
Report findings as:
- ✅ PASS: {description}
- ❌ FAIL: {description} — {what to fix}
- ⚠️ WARN: {description} — {suggestion}
