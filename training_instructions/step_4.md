## Step 4 — Reward function decomposition

**Objective:** Expose each of the 5 grader components as a separate TRL-compatible reward function. Do **not** pass a single aggregated reward.

**Context:**
- TRL reward functions have signature `def reward_fn(completions, **kwargs) -> list[float]`. The kwargs are the extra fields the rollout function returned.
- Each reward function here is a one-line passthrough: pull its component from kwargs and return it as a list of floats. The rollout already did the actual scoring (via `RewardComputer.grader_score` and its sub-score helpers) — these are just adapters.
- Component names must exactly match the keys produced in Step 3.

**Deliverable:** `training/rewards.py` exporting 5 functions and a `REWARD_FUNCS` list:

| Function | Pulls from kwargs key | Grader weight |
|----------|----------------------|---------------|
| `reward_coverage` | `coverage_reward` | 0.35 |
| `reward_ssr_integrity` | `ssr_integrity_reward` | 0.20 |
| `reward_cabin_match` | `cabin_match_reward` | 0.15 |
| `reward_group_integrity` | `group_integrity_reward` | 0.15 |
| `reward_deadline` | `deadline_reward` | 0.15 |

`REWARD_FUNCS` aggregates these 5 callables in the order matching the existing grader weights. The corresponding weights list (also exported from this module) is `[0.35, 0.20, 0.15, 0.15, 0.15]`, kept in lockstep with `REWARD_FUNCS` so TRL's `reward_weights` arg can be passed straight through.

**Constraints:**
- No grading logic in this file — that lives in the env's `RewardComputer` and is invoked from Step 3's rollout. These functions are adapters only.
- Default to 0.0 if a component key is missing (defensive — early training can have malformed trajectories).

**Definition of done:** Each reward function returns a list whose length matches `completions` length. Importing `REWARD_FUNCS` and iterating returns 5 callables. Importing the weights list returns a length-5 list summing to 1.0.
