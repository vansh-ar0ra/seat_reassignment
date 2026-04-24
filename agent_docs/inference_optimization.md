# Inference Optimization: Batch-Plan Pipeline

## Problem

The current `inference.py` uses a naive one-LLM-call-per-step loop. Every action — including predictable survey calls like `list_passengers` — requires a full LLM round-trip. For a hard-tier episode with 25 passengers, this means ~50+ LLM calls, most of which are wasted on decisions the code could make deterministically.

---

## Solution: Three-Phase Pipeline

Replace the call-per-step loop with a structured pipeline that separates **deterministic work** (survey, execution) from **reasoning work** (planning).

### Phase 1: Survey (deterministic — no LLM)

Hard-code the information-gathering steps. The inference loop executes these unconditionally before ever calling the LLM.

**Step sequence:**

```
1. list_passengers()              → get summary of all passengers
2. list_alternative_flights()     → get all flights with availability
3. For each passenger where has_ssr=True OR has_deadline=True OR group_id is not null:
     get_passenger_details(pid)   → get full constraint details
```

**Why these passengers only?**

`list_passengers` already returns `priority_tier`, `group_id`, `has_ssr`, `has_deadline`, and `booked` status for everyone. Unconstrained passengers (no SSR, no deadline, no group) only need their `original_cabin`, which can also be fetched — but in practice the LLM can infer the booking from the summary alone. We fetch details for constrained passengers because the LLM needs their exact SSR flags, deadline times, group integrity type, and group size to make correct decisions.

**Step budget consumed:**

| Tier | list_pax | list_flights | get_details | Total survey steps |
|------|----------|-------------|-------------|-------------------|
| Easy | 1 | 1 | 0 (no constraints) | 2 |
| Medium | 1 | 1 | ~7 (2 SSR + 2 deadline + 5 group members, some overlap) | ~9 |
| Hard | 1 | 1 | ~16 (6 SSR + 5 deadline + 10 group members, some overlap) | ~14 |

After Phase 1, the LLM receives ALL necessary data in a single context window.

### Phase 2: Plan (single LLM call)

Call the LLM once with the complete survey data. Ask it to return a **full booking plan** as a JSON array of actions:

```json
[
  {"tool_name": "book_group", "args": {"group_id": "GRP-M01", "flight_id": "FL-201", "cabin_assignments": {"PAX-M003": "economy", "PAX-M004": "economy", "PAX-M005": "economy"}}},
  {"tool_name": "book_passenger", "args": {"passenger_id": "PAX-M001", "flight_id": "FL-201", "cabin": "business"}},
  ...
  {"tool_name": "finalize_plan", "args": {}}
]
```

**Prompt instructions for the LLM:**

1. Book hard groups first using `book_group` (atomic, same flight).
2. Book SSR + deadline passengers next (most constrained).
3. Book soft groups using `book_group` where possible.
4. Book remaining passengers by priority tier descending.
5. Match original cabin. If unavailable, upgrade before downgrade.
6. End with `finalize_plan`.
7. Return the entire plan as a single JSON array.

**LLM decision: `book_group` vs `book_passenger`**

The LLM decides based on what it learned from `list_passengers`:

- If `group_id` is present AND group is "hard" → must use `book_group`
- If `group_id` is present AND group is "soft" → prefer `book_group`, fall back to individual if capacity forces splitting
- If no `group_id` → use `book_passenger`

This decision is made during planning, not per-step.

### Phase 3: Execute + Recover (deterministic, LLM only on failure)

The inference loop drains the plan queue one `step()` at a time:

```python
for action in plan:
    result = env.step(action)
    if result.observation.tool_result["status"] == "error":
        # Collect error, break out, re-plan
        break
```

**Recovery (LLM call only if needed):**

If a booking fails (capacity exhausted, SSR mismatch the LLM didn't account for, etc.):

1. Call `list_alternative_flights()` to refresh availability.
2. Call LLM again with:
   - The remaining unbooked passengers
   - The error message
   - Updated flight availability
3. LLM returns a revised plan for the remaining passengers.
4. Resume execution.

In the happy path, **zero recovery calls are needed**. The LLM plans correctly on the first try because it had all the data.

---

## Implementation Plan for `inference.py`

### Functions to add

#### `run_survey(env) -> dict`

Executes Phase 1 deterministically. Returns a dict:

```python
{
    "passengers_summary": [...],        # from list_passengers
    "passenger_details": {pid: {...}},   # from get_passenger_details (constrained only)
    "flights": [...],                    # from list_alternative_flights
    "groups": {gid: [pids]},            # derived from passengers_summary
    "constrained_pids": [pid, ...],     # passengers with SSR/deadline/group
}
```

#### `build_plan_prompt(survey_data) -> str`

Formats the survey data into a user message for the LLM. Includes:

- Full passenger list with details for constrained passengers
- Full flight list with availability and SSR support
- Explicit instruction to return a JSON array of actions
- Priority ordering rules

#### `parse_plan_response(response_text) -> list[dict]`

Parses the LLM response into a list of action dicts. Handles:

- JSON array directly
- Markdown-wrapped JSON
- Fallback: extract individual JSON objects if array parse fails

#### `run_plan_execution(env, plan) -> tuple[list, FlightRebookingObservation | None]`

Drains the plan queue via `step()` calls. Returns:

- List of executed results (for logging)
- The observation that caused a failure (or None if all succeeded)

#### `build_recovery_prompt(error_obs, remaining_passengers, updated_flights) -> str`

Formats a recovery prompt for the LLM with:

- What went wrong
- Which passengers still need booking
- Current flight availability

### Modified functions

#### `run_task()` — rewrite

Replace the per-step LLM loop with:

```python
async def run_task(task_name, task_id, max_steps, client):
    env = FlightRebookingEnv(base_url=ENV_URL)
    result = await env.reset(task_id=task_id)
    obs = result.observation

    # Phase 1: Survey (deterministic)
    survey_data = await run_survey(env)

    # Phase 2: Plan (single LLM call)
    plan_prompt = build_plan_prompt(survey_data)
    plan = get_plan_from_llm(client, plan_prompt)

    # Phase 3: Execute
    results, error_obs = await run_plan_execution(env, plan)

    # Recovery loop (if needed, max 2 retries)
    retries = 0
    while error_obs and retries < 2:
        retries += 1
        # Refresh availability
        refresh_obs = await env.step(FlightRebookingAction(
            tool_name="list_alternative_flights", args={}
        ))
        # Get remaining passengers
        remaining = [pid for pid in survey_data["passengers_summary"]
                     if pid not in booked_set]
        # Re-plan
        recovery_prompt = build_recovery_prompt(error_obs, remaining, refresh_obs)
        revised_plan = get_plan_from_llm(client, recovery_prompt)
        results, error_obs = await run_plan_execution(env, revised_plan)

    # Finalize if not already done
    if not obs.done:
        await env.step(FlightRebookingAction(tool_name="finalize_plan", args={}))
```

#### `get_agent_action()` — remove

No longer needed. Replaced by `get_plan_from_llm()` which returns a full plan.

#### `fallback_action()` — simplify

Only used during recovery. Returns `finalize_plan` as last resort.

### System prompt changes

The `SYSTEM_PROMPT` needs a new section instructing the LLM to return a full plan:

```
OUTPUT FORMAT:
Return your complete booking plan as a JSON array of tool calls.
Each element is {"tool_name": "...", "args": {...}}.
The array should contain ALL bookings needed, followed by finalize_plan as the last element.
Do NOT return one action at a time — return the entire plan at once.
```

The strategy section should emphasize:

1. Hard groups first (use book_group)
2. SSR + deadline passengers next
3. Soft groups (use book_group where possible)
4. Remaining passengers by priority tier
5. Always end with finalize_plan

---

## Expected Performance

### LLM calls per episode

| Tier | Current (per-step) | Optimized (batch) | Savings |
|------|-------------------|-------------------|---------|
| Easy | ~15-20 | 1 (+ 0 recovery) | ~95% |
| Medium | ~30-40 | 1-2 (+ 0-1 recovery) | ~95% |
| Hard | ~50-70 | 1-3 (+ 0-2 recovery) | ~95% |

### Step budget usage

| Tier | Survey steps | Booking steps | Finalize | Total | Budget |
|------|-------------|--------------|----------|-------|--------|
| Easy | 2 | 8 | 1 | 11 | 30 |
| Medium | ~9 | 12-15 | 1 | ~23 | 60 |
| Hard | ~14 | 21-25 | 1 | ~40 | 90 |

All tiers have comfortable headroom for recovery retries.

### Quality impact

- **Better decisions**: LLM sees ALL data at once instead of incrementally. Can plan globally optimal assignments.
- **Fewer errors**: No partial-information mistakes (e.g., booking a passenger on a flight then realizing SSR doesn't match).
- **Consistent**: Deterministic survey means the LLM always gets the same data format.

---

## Risk Areas

1. **LLM response parsing**: The LLM might not return a clean JSON array. `parse_plan_response` must be robust — handle markdown wrapping, partial arrays, and fallback extraction.

2. **book_group args complexity**: `cabin_assignments` is a nested dict. Some LLMs struggle with nested JSON in arrays. The prompt must include clear examples.

3. **Context length**: For hard tier, the survey data + prompt could be large. Keep the formatting concise — don't dump raw JSON, format as structured text.

4. **Recovery divergence**: If the LLM's revised plan keeps failing, cap retries at 2 and finalize with whatever is booked. Partial coverage is better than step-limit timeout.

5. **Passenger details for unconstrained passengers**: The current plan only fetches details for constrained passengers. If the LLM needs `original_cabin` for unconstrained passengers to decide cabin assignments, we have two options:
   - Option A: Fetch details for ALL passengers during survey (more steps, but complete data)
   - Option B: Include `original_cabin` in the `list_passengers` summary by modifying the tool
   
   **Recommendation**: Option A for now (fetch all). The step budget is generous enough. We can optimize to Option B later if needed.

---

## File Changes

| File | Change |
|------|--------|
| `inference.py` | Complete rewrite of run_task loop, new helper functions, updated system prompt |
| `agent_docs/inference_optimization.md` | This file (planning doc) |

No changes needed to environment, tools, rewards, models, or tests.
