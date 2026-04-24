---
  Two Layers of Reward

  The system has two independent layers that serve different purposes:

  Layer 1: Step Rewards (per action, during episode)

  These are immediate feedback signals returned after every step(). They accumulate in ep.cumulative_reward. The LLM sees them in the observation.

  Layer 2: Grader Score (terminal, at episode end)

  A single 0-1 score computed once when the episode ends. This is the hackathon evaluation metric — the number that actually matters for ranking. It's
  computed from 5 independent sub-scores.

  ---
  Layer 1: Step Rewards in Detail

  Information tool rewards

  The environment rewards gathering information but punishes redundancy:

  ┌────────────────────────────────────────────────────┬────────┬───────────────────────────────────────────────────────────────────────────────────┐
  │                       Action                       │ Reward │                          Logic (in reward_for_info_call)                          │
  ├────────────────────────────────────────────────────┼────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ list_passengers() first time                       │ +0.02  │ Encourages the agent to survey first                                              │
  ├────────────────────────────────────────────────────┼────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ list_passengers() 5th+ time with no bookings in    │ -0.01  │ Punishes spinning — the agent is calling list_passengers repeatedly without       │
  │ between                                            │        │ making progress                                                                   │
  ├────────────────────────────────────────────────────┼────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ get_passenger_details(pid) for unbooked pax        │ +0.02  │ Good — learning about someone you still need to book                              │
  ├────────────────────────────────────────────────────┼────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ get_passenger_details(pid) for already-booked pax  │ -0.01  │ Wasteful — you already dealt with them                                            │
  ├────────────────────────────────────────────────────┼────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ list_alternative_flights()                         │ +0.01  │ Always slightly useful since availability changes after bookings                  │
  ├────────────────────────────────────────────────────┼────────┼───────────────────────────────────────────────────────────────────────────────────┤
  │ get_flight_details(fid)                            │ +0.01  │ Gathering info before committing                                                  │
  └────────────────────────────────────────────────────┴────────┴───────────────────────────────────────────────────────────────────────────────────┘

  The churn detection at rewards.py:98 checks: has list_passengers been called 5+ times AND the last booking happened before the current run of
  list_passengers calls? If so, the agent is stuck in a loop.

  Booking rewards

  When book_passenger succeeds (environment.py:250-254), the reward depends on cabin match and priority weight:

  reward = base_reward × priority_weight(passenger.priority_tier)

  Where priority_weight maps: tier 1→1.5, tier 2→1.3, tier 3→1.0, tier 4→0.8, tier 5→0.6

  The base reward is determined by comparing assigned cabin to original cabin using _CABIN_RANK = {economy: 0, premium_economy: 1, business: 2}:

  ┌────────────┬──────┬─────────────────────┬────────────────────┐
  │  Outcome   │ Base │       Example       │   Tier 1 actual    │
  ├────────────┼──────┼─────────────────────┼────────────────────┤
  │ Same cabin │ 0.15 │ business → business │ 0.15 × 1.5 = 0.225 │
  ├────────────┼──────┼─────────────────────┼────────────────────┤
  │ Upgrade    │ 0.10 │ economy → business  │ 0.10 × 1.5 = 0.15  │
  ├────────────┼──────┼─────────────────────┼────────────────────┤
  │ Downgrade  │ 0.02 │ business → economy  │ 0.02 × 1.5 = 0.03  │
  └────────────┴──────┴─────────────────────┴────────────────────┘

  Then if the passenger had a downstream_deadline and the tool_result says deadline_met: true, an additional +0.05 × priority_weight is added
  (rewards.py:159-161).

  Group booking rewards

  reward_for_group_booking (rewards.py:165-208) works similarly but iterates over each member in the booked group, summing up individual cabin-match
  rewards. It also checks whether all members ended up in the same cabin (reason string changes but reward calculation is per-member).

  The deadline bonus for groups checks pax.get("downstream_deadline") — if any member has a deadline and the group was booked (on a flight that by
  definition meets it, since validation would have rejected otherwise), they get the bonus.

  Failure rewards

  ┌───────────────────────────────────────┬────────┬─────────────────────────────────────────────┐
  │                Outcome                │ Reward │                    Code                     │
  ├───────────────────────────────────────┼────────┼─────────────────────────────────────────────┤
  │ Failed booking (any validation error) │ -0.02  │ reward_for_failed_action                    │
  ├───────────────────────────────────────┼────────┼─────────────────────────────────────────────┤
  │ Unknown tool name                     │ -0.05  │ reward_for_invalid_tool                     │
  ├───────────────────────────────────────┼────────┼─────────────────────────────────────────────┤
  │ finalize_plan                         │ 0.0    │ No step reward — grading happens separately │
  └───────────────────────────────────────┴────────┴─────────────────────────────────────────────┘

  How step rewards flow

  In environment.py:317:
  ep.cumulative_reward += reward

  Every observation includes cumulative_reward and reward_reason so the LLM can see what it earned and why.

  ---
  Layer 2: Grader Score in Detail

  Computed at rewards.py:225-255 when the episode ends (finalize, step limit, or all-booked). The environment calls it at environment.py:296-301.

  The 5 sub-scores

  1. Coverage (weight 0.35) — terminal_breakdown, line 271
  coverage_score = n_booked / n_total
  Simple fraction. 8/8 = 1.0, 4/8 = 0.5. This is the most heavily weighted because getting passengers on flights is the primary goal.

  2. Cabin Match (weight 0.15) — _cabin_match_score, lines 303-318
  for each passenger:
      weight = priority_weight(tier)
      total_weight += weight
      if booked AND booked_cabin == original_cabin:
          matched_weight += weight
  score = matched_weight / total_weight
  Priority-weighted. A tier-1 passenger in the wrong cabin hurts more than a tier-5 passenger. Unbooked passengers count in the denominator
  (total_weight includes everyone), so they implicitly lower this score.

  3. Group Integrity (weight 0.15) — _group_integrity_score, lines 320-379

  For each group, scores:

  ┌──────────────────────────────────────────────┬─────────────────────────────────┐
  │                  Situation                   │              Score              │
  ├──────────────────────────────────────────────┼─────────────────────────────────┤
  │ All members on same flight, same cabin       │ 1.0                             │
  ├──────────────────────────────────────────────┼─────────────────────────────────┤
  │ All members on same flight, different cabins │ 0.7                             │
  ├──────────────────────────────────────────────┼─────────────────────────────────┤
  │ Hard group split across flights              │ 0.0 + counted as hard violation │
  ├──────────────────────────────────────────────┼─────────────────────────────────┤
  │ Soft group split across flights              │ 0.4                             │
  ├──────────────────────────────────────────────┼─────────────────────────────────┤
  │ Partially booked group (some unbooked)       │ Treated as split                │
  ├──────────────────────────────────────────────┼─────────────────────────────────┤
  │ No members booked                            │ 0.0, no hard violation          │
  ├──────────────────────────────────────────────┼─────────────────────────────────┤
  │ No groups exist at all                       │ 1.0 (nothing to violate)        │
  └──────────────────────────────────────────────┴─────────────────────────────────┘

  The score is averaged across all groups. Hard violations are counted separately.

  4. Deadline (weight 0.15) — _deadline_score, lines 381-403
  for each passenger WITH a deadline:
      weight = priority_weight(tier)
      total_weight += weight
      if booked AND flight.arrival_time <= deadline:
          met_weight += weight
  score = met_weight / total_weight
  Only counts passengers who have deadlines. If no one has a deadline, returns 1.0. Uses meets_deadline() from tools.py to compare HH:MM strings.

  5. SSR Integrity (weight 0.20) — _ssr_integrity_score, lines 405-432
  for each passenger with SSR flags who is booked:
      if passenger's required SSRs NOT subset of flight's supported SSRs:
          violations += 1
  score = max(0.0, 1.0 - 0.25 × violations)
  Each violation costs 0.25 from a starting 1.0. So 4+ violations = 0.0. Violations are also counted as hard violations.

  Final assembly (grader_score, lines 243-255)

  score = (0.35 × coverage) + (0.15 × cabin_match) + (0.15 × group_integrity)
        + (0.15 × deadline) + (0.20 × ssr_integrity)

  # Hard constraint penalties: -0.15 per violation
  score -= 0.15 × hard_violations

  # Clamp to (EPS, 1-EPS) where EPS = 0.0001
  score = max(0.0001, min(0.9999, score))

  The hard violation penalty is on top of the sub-score damage. For example, a hard group split across flights gets 0.0 in the group integrity sub-score
   AND -0.15 from the penalty. This is intentional — hard constraints should hurt badly.

  What triggers terminal grading

  In environment.py:290-315, grading fires when done = True, which happens if:
  1. finalize_plan() was called
  2. step_count >= max_steps (timeout)
  3. All passengers booked (auto-finalize)

  The terminal_breakdown dict is attached to the final tool_result so the LLM (and tests) can see all sub-scores.

  ---
  Example Walkthrough

  Medium tier: 15 passengers, tier weights sum ≈ 15.6

  Perfect run: all 15 booked, same cabin, groups together, all deadlines met, no SSR violations:
  - coverage = 15/15 = 1.0
  - cabin_match = 15.6/15.6 = 1.0
  - group_integrity = avg(1.0, 1.0) = 1.0
  - deadline = met/met = 1.0
  - ssr_integrity = 1.0
  - grader = 0.35 + 0.15 + 0.15 + 0.15 + 0.20 = 1.0 → clamped to 0.9999

  Worst case: 0 booked:
  - coverage = 0.0
  - cabin_match = 0.0
  - group_integrity = 0.0 (unbooked groups)
  - deadline = 0.0 (unbooked deadline pax) ... actually returns 1.0 if denominator is 0 — wait, no, deadline passengers exist but aren't booked, so
  met_weight=0, total_weight>0 → score=0.0
  - ssr_integrity = 1.0 (no booked SSR pax → no violations possible)
  - grader = 0 + 0 + 0 + 0 + 0.20 = 0.20 → stays 0.20

  That 0.20 floor (from SSR integrity being trivially 1.0 with no bookings) is why the test_zero_coverage test expects GRADER_W_SSR_INTEGRITY * 1.0
  rather than EPS.