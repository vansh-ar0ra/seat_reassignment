## Step 5 — Dataset of task seeds

**Objective:** Build the prompt dataset that drives episode sampling. Each "prompt" is a seed for the env to generate a scenario from.

**Context:**
- Env uses procedural generation — seed determines scenario. We need ~250-500 seeds spread across easy/medium/hard tiers.
- For curriculum learning later, tag each seed with its tier in the dataset so we can filter at training time.
- The actual prompt text the model sees is constructed inside the rollout function from the env reset; the dataset entry just needs to carry the seed and tier.

**Deliverable:** `training/dataset.py` with `build_dataset(n_easy, n_medium, n_hard, base_seed=0) -> Dataset` returning a HuggingFace `Dataset` with columns `prompt` (string framing the disruption response task — same for all rows, satisfies TRL's interface) and `seed` and `tier`.

**Constraints:**
- Use deterministic seed assignment so dataset is reproducible.
- Default config: 50 easy, 100 medium, 100 hard.

**Definition of done:** `build_dataset()` returns a `Dataset` of correct length with all three tiers represented and unique seeds.

