# Testing Strategy: Flight Rebooking Environment

## Principles

1. Tests instantiate the environment directly — no server, no WebSocket
2. Every test tier has a known-optimal assignment that achieves grader ≈ 1.0
3. Tests verify both happy paths and error paths
4. Reward constants are imported and tested against expected values
5. All 3 tiers (easy, medium, hard) have dedicated test classes

---

## Test File: `tests/test_environment.py`

### Helpers (reuse across tests)
```python
def make_env(task_id="medium"):
    env = FlightRebookingEnvironment()
    env.reset(task_id=task_id)
    return env

def call_tool(env, tool_name, **args):
    return env.step(FlightRebookingAction(tool_name=tool_name, args=args))
```

### Required Test Classes

#### 1. TestReset
- `test_basic_fields` — done=False, reward=0, passengers_remaining > 0, step_count=0
- `test_no_bookings_initially` — booked_summary is empty
- `test_max_steps_set` — matches config
- `test_unknown_task_raises` — ValueError for bad task_id

#### 2. TestListPassengers
- `test_returns_all_passengers` — count matches data
- `test_returns_summary_not_full_details` — has passenger_id, priority_tier, group_id, has_preference; does NOT have original_cabin or ssr_flags details
- `test_reward_is_small_positive_first_call` — +0.02 first time

#### 3. TestGetPassengerDetails
- `test_returns_full_record` — all fields present including paid_window, paid_legroom
- `test_nonexistent_passenger_errors` — status=error
- `test_already_booked_passenger_penalty` — small negative reward

#### 4. TestListAlternativeFlights
- `test_returns_all_flights` — count matches data
- `test_availability_decrements_after_booking` — book a passenger, re-call, count reduced
- `test_includes_ssr_support` — supports_ssr field present
- `test_includes_seat_features` — seat_features field present with window/legroom counts per cabin
- `test_seat_features_decrement_after_preference_booking` — book passenger with paid_window, re-call, window count reduced

#### 5. TestGetFlightDetails
- `test_returns_full_flight` — all fields present
- `test_nonexistent_flight_errors` — status=error

#### 6. TestBookPassenger
- `test_successful_booking` — status=success, passengers_remaining decremented
- `test_cabin_availability_decremented` — verify via get_flight_details after booking
- `test_same_cabin_positive_reward` — booking into original cabin → positive reward
- `test_upgrade_positive_reward` — economy→business → positive reward (smaller than same-cabin)
- `test_downgrade_small_reward` — business→economy → small positive reward
- `test_double_booking_errors` — booking already-booked passenger → error
- `test_no_availability_errors` — booking into full cabin → error
- `test_ssr_mismatch_errors` — passenger with UM SSR on flight that doesn't support UM → error
- `test_deadline_violation_errors` — passenger with deadline booked on late flight → error
- `test_nonexistent_passenger_errors`
- `test_nonexistent_flight_errors`
- `test_invalid_cabin_errors`
- `test_preference_satisfied_when_available` — passenger with paid_window booked on cabin with window > 0 → preferences_satisfied includes "window"
- `test_preference_not_satisfied_when_unavailable` — passenger with paid_window booked on cabin with window = 0 → preferences_satisfied does not include "window"
- `test_preference_bonus_reward` — booking passenger with satisfied preference gives bonus step reward

#### 7. TestBookGroup
- `test_successful_group_booking` — all members booked atomically
- `test_atomic_failure` — if one member can't fit, NONE are booked
- `test_hard_group_same_flight` — all on same flight (by design)
- `test_split_cabin_allowed` — different cabin_assignments per member on same flight
- `test_ssr_check_all_members` — if ANY member has unsupported SSR → all fail
- `test_nonexistent_group_errors`
- `test_partially_booked_group_errors` — if some members already booked → error
- `test_group_booking_preference_satisfaction` — group members with paid preferences get them satisfied when available

#### 8. TestFinalizePlan
- `test_triggers_done` — done=True after finalize
- `test_grader_score_present` — tool_result has grader_score
- `test_grader_score_in_range` — EPS <= score <= 1-EPS
- `test_step_after_done_raises` — RuntimeError

#### 9. TestEasyTask
- `test_reset_fields` — correct passenger count
- `test_full_optimal_booking` — book all passengers optimally → grader ≈ 1.0
- `test_no_groups_no_ssr` — easy data has no groups, no SSR
- `test_step_limit_terminates`

#### 10. TestMediumTask
- `test_reset_fields` — correct passenger count
- `test_group_booking_works` — can book hard group
- `test_ssr_respected` — SSR passengers only on compatible flights
- `test_deadline_respected` — deadline passengers arrive on time
- `test_full_optimal_booking` — grader ≈ 1.0 with optimal assignments

#### 11. TestHardTask
- `test_reset_fields` — correct passenger count
- `test_multiple_groups` — hard + soft groups
- `test_ssr_scarcity` — not all flights support all SSRs
- `test_capacity_pressure` — some cabins at limit
- `test_full_optimal_booking` — grader ≈ 1.0 with optimal assignments

#### 12. TestInvalidTool
- `test_unknown_tool_error` — status=error, negative reward
- `test_empty_args_handled` — doesn't crash

---

## Test File: `tests/test_rewards.py`

### Required Test Classes

#### 1. TestInfoCallRewards
- `test_first_list_passengers_positive`
- `test_repeated_list_passengers_negative` (5th+ with no bookings)
- `test_get_details_unbooked_positive`
- `test_get_details_booked_negative`
- `test_list_flights_always_small_positive`

#### 2. TestBookingRewards
- `test_same_cabin_high_reward`
- `test_upgrade_medium_reward`
- `test_downgrade_small_reward`
- `test_split_cabin_fallback_reward`
- `test_priority_weight_scaling` — Tier 1 gets higher reward than Tier 5 for same outcome
- `test_deadline_met_bonus`
- `test_preference_satisfaction_bonus` — each satisfied preference adds +0.03 × priority_weight
- `test_hard_constraint_violation_heavy_penalty`
- `test_failed_booking_small_penalty`

#### 3. TestGraderScore
- `test_perfect_score_all_booked_same_cabin` — ≈ 1.0
- `test_zero_coverage` — very low score
- `test_partial_coverage` — between 0 and 1
- `test_ssr_violation_penalizes` — drops score significantly
- `test_group_split_penalizes` — hard group split → score drops
- `test_deadline_missed_penalizes`
- `test_preference_missed_penalizes` — unsatisfied paid preferences lower score
- `test_grader_is_deterministic` — same input → same output
- `test_grader_clamped_to_eps_range`

#### 4. TestPriorityWeights
- `test_tier1_highest`
- `test_tier5_lowest`
- `test_unknown_tier_defaults_to_1`

#### 5. TestCabinMatchScore
- `test_all_matched`
- `test_none_matched`
- `test_partial_match_priority_weighted`

#### 6. TestGroupIntegrityScore
- `test_all_same_flight_same_cabin` — 1.0
- `test_same_flight_diff_cabin` — 0.7
- `test_hard_group_split_flights` — 0.0
- `test_soft_group_split_flights` — 0.4
- `test_no_groups` — 1.0 (no groups to violate)

#### 7. TestPreferenceScore
- `test_all_preferences_satisfied` — 1.0
- `test_no_preferences_satisfied` — 0.0
- `test_partial_preferences_priority_weighted` — higher tier preferences weigh more
- `test_no_passengers_have_preferences` — 1.0 (nothing to violate)
- `test_mixed_window_legroom` — passenger with both preferences, only one satisfied → 0.5 weighted

---

## Data Requirements for Tests

Each tier's data must include a **documented optimal assignment** — a dict mapping `{passenger_id: (flight_id, cabin)}` that achieves grader_score ≈ 1.0. This is defined as a constant in the test file and used to run the full-episode test.

Example:
```python
OPTIMAL_EASY = {
    "PAX-E001": ("FL-201", "business"),
    "PAX-E002": ("FL-201", "business"),
    "PAX-E003": ("FL-201", "economy"),
    ...
}
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Just environment tests
pytest tests/test_environment.py -v

# Just reward tests  
pytest tests/test_rewards.py -v

# Single test class
pytest tests/test_environment.py::TestBookPassenger -v
```
