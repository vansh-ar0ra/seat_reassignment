# Training Plan: SFT + GRPO for Flight Rebooking Agent

## Overview

Two-phase training pipeline to teach an LLM to solve the flight rebooking environment:

1. **Phase 1 — SFT (Supervised Fine-Tuning)**: Teach the model the tool-calling format, basic strategy, and constraint awareness using expert trajectories.
2. **Phase 2 — GRPO (Group Relative Policy Optimization)**: Reinforce the model against the live environment using the grader score as reward, so it learns trade-off reasoning that can't be captured in static demonstrations.

**Base model**: `Qwen/Qwen2.5-7B-Instruct` (already instruction-tuned, supports tool calling)
**Library**: `trl >= 1.2.0` (HuggingFace TRL)
**Hardware target**: 1-2 nodes with 4-8 GPUs (A100 40GB or equivalent)

---

## Phase 1: SFT — Learning the Tool-Calling Protocol

### Goal

Teach the model to:
- Output valid JSON tool calls (`{"tool_name": "...", "args": {...}}`)
- Follow a sensible booking strategy (survey first, constrained passengers first, finalize when done)
- Respect hard constraints (SSR, deadlines, hard groups)
- Understand the observation format (state, tool_result, reward)

### 1.1 Data Collection

**Source**: Run the environment with an expert policy (scripted solver) across many seeds.

Create a script `training/collect_sft_data.py` that:

```
for seed in range(N_EPISODES):
    for difficulty in [0.2, 0.5, 0.8]:
        1. env.reset(seed=seed, task_id=f"seed_{seed}")
        2. Run expert_policy(env) which uses a greedy-optimal solver:
           - list_passengers -> list_alternative_flights
           - Sort passengers by: hard constraints first, then priority tier
           - For each passenger, find best valid flight/cabin (same cabin > upgrade > downgrade)
           - Use book_group for groups, book_passenger for individuals
           - Handle errors by trying alternatives
           - finalize_plan
        3. Record each (observation, action) pair as a conversation turn
```

**Target**: 5,000-10,000 expert episodes across difficulties.

### 1.2 Dataset Format

TRL SFTTrainer expects **conversational format** with `messages` column:

```python
{
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Task: A flight has been cancelled...\n\n=== Step 0/35 | Booked: 0/15 | ..."},
        {"role": "assistant", "content": '{"tool_name": "list_passengers", "args": {}}'},
        {"role": "user", "content": "Last tool result: {\"status\": \"success\", ...}\n\n=== Step 1/35 | ..."},
        {"role": "assistant", "content": '{"tool_name": "list_alternative_flights", "args": {}}'},
        # ... all turns in the episode ...
        {"role": "assistant", "content": '{"tool_name": "finalize_plan", "args": {}}'},
    ]
}
```

Each episode becomes one row in a HuggingFace `Dataset`. The system prompt is the same `SYSTEM_PROMPT` from `inference.py`.

### 1.3 Data Processing Script

Create `training/build_sft_dataset.py`:

```python
from datasets import Dataset

def build_dataset(episodes_dir: str) -> Dataset:
    """
    Load collected episodes from JSON files and convert to
    conversational format for SFTTrainer.
    
    Each episode JSON has:
    {
        "task_id": "seed_42",
        "difficulty": 0.5,
        "score": 0.95,
        "turns": [
            {"observation_text": "...", "action": {"tool_name": "...", "args": {...}}},
            ...
        ]
    }
    """
    rows = []
    for episode_file in sorted(Path(episodes_dir).glob("*.json")):
        episode = json.load(open(episode_file))
        
        # Only use high-quality episodes (score > 0.8)
        if episode["score"] < 0.8:
            continue
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in episode["turns"]:
            messages.append({"role": "user", "content": turn["observation_text"]})
            messages.append({"role": "assistant", "content": json.dumps(turn["action"])})
        
        rows.append({"messages": messages})
    
    return Dataset.from_list(rows)
```

### 1.4 SFT Training Script

Create `training/train_sft.py`:

```python
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

dataset = load_from_disk("training/sft_dataset")

# LoRA for memory efficiency
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="checkpoints/sft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,           # Higher LR for LoRA
    warmup_ratio=0.05,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    
    # SFT-specific
    max_length=4096,              # Episodes can be long (multi-turn)
    assistant_only_loss=True,     # Only train on assistant (tool call) tokens
    packing=False,                # Don't pack — episodes are multi-turn conversations
)

trainer = SFTTrainer(
    model=MODEL_NAME,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)
trainer.train()
trainer.save_model("checkpoints/sft/final")
```

**Key decisions**:
- `assistant_only_loss=True` — only compute loss on the JSON tool-call tokens, not the observation text
- `max_length=4096` — medium/hard episodes can be 30+ turns; 4096 tokens should cover most
- LoRA rank 32 — good balance of capacity vs memory for 7B model
- 3 epochs — sufficient for format learning without overfitting the strategy

### 1.5 SFT Data Augmentation Strategies

- **Difficulty mixing**: 40% easy, 35% medium, 25% hard episodes
- **Error recovery trajectories**: Include episodes where the expert makes a deliberate mistake (e.g., books wrong cabin), gets the error, then corrects — teaches the model to recover
- **Suboptimal demonstrations**: Include some episodes with score 0.7-0.8 (not perfect) so the model sees realistic imperfect states
- **Event handling**: Include procedural episodes (difficulty >= 0.3) where mid-episode events fire, so the model sees unbook/rebook patterns

---

## Phase 2: GRPO — Learning Trade-Off Reasoning

### Goal

The SFT model knows the format and basic strategy. GRPO teaches it:
- When to downgrade a gold member vs. leave them unbooked
- When to upgrade a low-priority passenger (if seat would go unused)
- How to handle Pareto-impossible constraint sets
- Cost-aware decision making under budget pressure
- Adapting to mid-episode events

### 2.1 Environment Wrapper for GRPO

TRL's GRPOTrainer supports `environment_factory` — we wrap our environment as a class whose public methods become tools.

Create `training/grpo_env.py`:

```python
import json
from server.environment import FlightRebookingEnvironment
from models import FlightRebookingAction

class FlightRebookingGRPOEnv:
    """
    GRPO-compatible environment wrapper.
    
    Public methods (except reset) are exposed as tools to the LLM.
    Each method must have type hints and Google-style docstrings.
    """
    
    def __init__(self):
        self._env = FlightRebookingEnvironment()
        self._obs = None
        self._done = False
        self.reward = 0.0
        self.grader_score = 0.0
    
    def reset(self, *, difficulty: float = 0.5, seed: int = 0, **kwargs) -> str:
        """Reset the environment. Called by GRPOTrainer before each generation."""
        import random
        if seed == 0:
            seed = random.randint(1, 100000)
        
        self._obs = self._env.reset(seed=seed, task_id=f"seed_{seed}")
        self._done = False
        self.reward = 0.0
        self.grader_score = 0.0
        
        # Return initial state as string appended to the user message
        return self._format_state(self._obs)
    
    def list_passengers(self) -> str:
        """
        List all passengers needing rebooking with summary info.
        
        Returns:
            JSON string with passenger summaries including ID, priority, group, SSR, deadline, loyalty.
        """
        return self._step("list_passengers", {})
    
    def get_passenger_details(self, passenger_id: str) -> str:
        """
        Get full details for one passenger.
        
        Args:
            passenger_id: The passenger ID (e.g., "PAX-001").
        
        Returns:
            JSON string with full passenger record.
        """
        return self._step("get_passenger_details", {"passenger_id": passenger_id})
    
    def list_alternative_flights(self) -> str:
        """
        List all available alternative flights with cabin availability and SSR support.
        
        Returns:
            JSON string with flight details.
        """
        return self._step("list_alternative_flights", {})
    
    def get_flight_details(self, flight_id: str) -> str:
        """
        Get details for one specific flight.
        
        Args:
            flight_id: The flight ID (e.g., "FL-201").
        
        Returns:
            JSON string with flight details and current availability.
        """
        return self._step("get_flight_details", {"flight_id": flight_id})
    
    def book_passenger(self, passenger_id: str, flight_id: str, cabin: str) -> str:
        """
        Book one passenger onto a flight in a specific cabin.
        
        Args:
            passenger_id: The passenger ID to book.
            flight_id: The target flight ID.
            cabin: The cabin class (economy, premium_economy, business).
        
        Returns:
            JSON string with booking result, cost, and cabin match info.
        """
        return self._step("book_passenger", {
            "passenger_id": passenger_id,
            "flight_id": flight_id,
            "cabin": cabin,
        })
    
    def book_group(self, group_id: str, flight_id: str, cabin_assignments: str) -> str:
        """
        Book an entire group onto one flight atomically.
        
        Args:
            group_id: The group ID (e.g., "GRP-001").
            flight_id: The target flight ID.
            cabin_assignments: JSON string mapping passenger_id to cabin for each member.
        
        Returns:
            JSON string with group booking result and total cost.
        """
        try:
            assignments = json.loads(cabin_assignments) if isinstance(cabin_assignments, str) else cabin_assignments
        except json.JSONDecodeError:
            return json.dumps({"status": "error", "message": "Invalid cabin_assignments JSON"})
        return self._step("book_group", {
            "group_id": group_id,
            "flight_id": flight_id,
            "cabin_assignments": assignments,
        })
    
    def unbook_passenger(self, passenger_id: str) -> str:
        """
        Remove an existing booking, freeing the seat back to inventory.
        
        Args:
            passenger_id: The passenger ID to unbook.
        
        Returns:
            JSON string with unbook result and cost reversed.
        """
        return self._step("unbook_passenger", {"passenger_id": passenger_id})
    
    def finalize_plan(self) -> str:
        """
        End the episode and trigger final grading.
        
        Returns:
            JSON string with grader score and terminal breakdown.
        """
        return self._step("finalize_plan", {})
    
    def _step(self, tool_name: str, args: dict) -> str:
        if self._done:
            return json.dumps({"status": "error", "message": "Episode already done"})
        
        obs = self._env.step(FlightRebookingAction(tool_name=tool_name, args=args))
        self._obs = obs
        self.reward += obs.reward
        
        if obs.done:
            self._done = True
            if obs.tool_result and "grader_score" in obs.tool_result:
                self.grader_score = obs.tool_result["grader_score"]
        
        return self._format_result(obs)
    
    def _format_state(self, obs) -> str:
        parts = [
            f"Step {obs.step_count}/{obs.max_steps} | "
            f"Booked: {obs.passengers_booked}/{obs.passengers_total} | "
            f"Remaining: {obs.passengers_remaining} | "
            f"Cost: ${obs.total_cost:.0f} (budget: ${obs.compensation_budget:.0f})"
        ]
        return "\n".join(parts)
    
    def _format_result(self, obs) -> str:
        parts = []
        if obs.tool_result:
            parts.append(json.dumps(obs.tool_result, indent=2))
        parts.append(self._format_state(obs))
        return "\n".join(parts)
```

### 2.2 Reward Functions

GRPO needs reward functions that score completed episodes. We use **two reward functions** combined:

```python
def grader_reward(environments, **kwargs):
    """Primary reward: the environment's grader score (0 to 1)."""
    return [env.grader_score for env in environments]

def efficiency_reward(environments, **kwargs):
    """Bonus for finishing with fewer steps."""
    rewards = []
    for env in environments:
        obs = env._obs
        if obs is None:
            rewards.append(0.0)
            continue
        # Bonus for using fewer steps (max 0.1 extra)
        step_ratio = obs.step_count / max(1, obs.max_steps)
        efficiency = max(0.0, 0.1 * (1.0 - step_ratio))
        rewards.append(efficiency)
    return rewards
```

### 2.3 Prompt Dataset

GRPO needs a dataset of prompts (the initial user message). We generate these from different seeds/difficulties:

Create `training/build_grpo_prompts.py`:

```python
from datasets import Dataset

def build_prompt_dataset(n_prompts: int = 5000) -> Dataset:
    """
    Build a dataset of initial prompts for GRPO training.
    
    Each prompt is a conversation start: system message + initial task description.
    The environment_factory handles the actual environment interaction.
    """
    rows = []
    for i in range(n_prompts):
        # Vary difficulty
        difficulty = [0.2, 0.4, 0.5, 0.6, 0.8][i % 5]
        seed = i + 1
        
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    "A flight has been cancelled. Rebook all passengers onto "
                    "alternative flights, respecting constraints and priorities.\n\n"
                    "Call list_passengers to begin."
                )},
            ],
            "difficulty": difficulty,
            "seed": seed,
        })
    
    return Dataset.from_list(rows)
```

### 2.4 GRPO Training Script

Create `training/train_grpo.py`:

```python
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from training.grpo_env import FlightRebookingGRPOEnv

MODEL_NAME = "checkpoints/sft/final"  # Start from SFT checkpoint

dataset = load_from_disk("training/grpo_prompts")

def grader_reward(environments, **kwargs):
    return [env.grader_score for env in environments]

def efficiency_reward(environments, **kwargs):
    rewards = []
    for env in environments:
        obs = env._obs
        if obs is None:
            rewards.append(0.0)
            continue
        step_ratio = obs.step_count / max(1, obs.max_steps)
        rewards.append(max(0.0, 0.1 * (1.0 - step_ratio)))
    return rewards

peft_config = LoraConfig(
    r=16,                   # Smaller rank for RL (less overfitting risk)
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

training_args = GRPOConfig(
    output_dir="checkpoints/grpo",
    
    # GRPO-specific
    num_generations=4,          # Generate 4 completions per prompt for advantage estimation
    max_completion_length=4096, # Full episode can be long
    
    # Training
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-7,         # Very low LR for RL fine-tuning
    warmup_ratio=0.05,
    bf16=True,
    gradient_checkpointing=True,
    
    # Logging
    logging_steps=5,
    save_strategy="steps",
    save_steps=200,
    log_completions=True,       # Log sample completions for debugging
    
    # GRPO algorithm
    beta=0.0,                   # No KL penalty (per recent best practices)
    scale_rewards=True,         # Normalize rewards within each group
    
    # vLLM for faster generation (if available)
    # use_vllm=True,
    # vllm_mode="colocate",
)

trainer = GRPOTrainer(
    model=MODEL_NAME,
    args=training_args,
    reward_funcs=[grader_reward, efficiency_reward],
    reward_weights=[1.0, 0.5],          # Grader score is primary
    train_dataset=dataset,
    peft_config=peft_config,
    environment_factory=FlightRebookingGRPOEnv,
)

trainer.train()
trainer.save_model("checkpoints/grpo/final")
```

**Key decisions**:
- `num_generations=4` — generate 4 episodes per prompt to compute group-relative advantage
- `beta=0.0` — no KL penalty (recent papers show this works fine, prevents mode collapse via clipping instead)
- `reward_weights=[1.0, 0.5]` — grader score is the primary signal, efficiency is secondary
- `learning_rate=5e-7` — very low to avoid catastrophic forgetting of SFT format knowledge
- LoRA rank 16 — smaller than SFT phase to reduce overfitting during RL
- Start from SFT checkpoint — not from base model

### 2.5 Alternative: GRPO Without environment_factory

If `environment_factory` doesn't fit (e.g., environment is remote), use a custom `reward_func` that runs episodes post-hoc:

```python
import asyncio
from client import FlightRebookingEnv
from models import FlightRebookingAction

async def run_episode_and_score(completion_text: str, seed: int, difficulty: float) -> float:
    """Run the completion's tool calls against the environment and return grader score."""
    env = FlightRebookingEnv(base_url="http://localhost:8000")
    try:
        result = await env.reset(task_id=f"seed_{seed}")
        
        # Parse the completion into a sequence of tool calls
        tool_calls = parse_tool_calls_from_completion(completion_text)
        
        for tool_call in tool_calls:
            if result.done:
                break
            result = await env.step(FlightRebookingAction(
                tool_name=tool_call["tool_name"],
                args=tool_call["args"],
            ))
        
        # Finalize if not done
        if not result.done:
            result = await env.step(FlightRebookingAction(
                tool_name="finalize_plan", args={}
            ))
        
        obs = result.observation
        if obs.tool_result and "grader_score" in obs.tool_result:
            return obs.tool_result["grader_score"]
        return 0.0
    finally:
        await env.close()


def environment_reward(completions, seed, difficulty, **kwargs):
    """Reward function that replays completions against the environment."""
    loop = asyncio.get_event_loop()
    scores = []
    for completion, s, d in zip(completions, seed, difficulty):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        score = loop.run_until_complete(run_episode_and_score(text, s, d))
        scores.append(score)
    return scores
```

---

## Phase 3: Files to Create

### Directory structure

```
training/
    collect_sft_data.py        # Run expert policy, save episode JSONs
    expert_policy.py           # Scripted optimal solver
    build_sft_dataset.py       # Convert episodes to HF Dataset
    train_sft.py               # SFT training script
    build_grpo_prompts.py      # Create prompt dataset for GRPO
    grpo_env.py                # Environment wrapper for GRPOTrainer
    train_grpo.py              # GRPO training script
    eval.py                    # Evaluate trained model against all tiers
    configs/
        sft_config.yaml        # Hyperparameters for SFT
        grpo_config.yaml       # Hyperparameters for GRPO
```

### Implementation Order

| Step | Script | Depends On | Output |
|------|--------|------------|--------|
| 1 | `expert_policy.py` | environment.py, tools.py | Greedy-optimal solver |
| 2 | `collect_sft_data.py` | expert_policy.py | `data/sft_episodes/*.json` |
| 3 | `build_sft_dataset.py` | collected episodes | `training/sft_dataset/` (HF Dataset) |
| 4 | `train_sft.py` | sft_dataset | `checkpoints/sft/final/` |
| 5 | `build_grpo_prompts.py` | inference.py (SYSTEM_PROMPT) | `training/grpo_prompts/` (HF Dataset) |
| 6 | `grpo_env.py` | environment.py, models.py | GRPO env wrapper class |
| 7 | `train_grpo.py` | SFT checkpoint, grpo_env, grpo_prompts | `checkpoints/grpo/final/` |
| 8 | `eval.py` | trained model | Score reports per difficulty |

---

## Hyperparameter Summary

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Base model | Qwen2.5-7B-Instruct | SFT checkpoint |
| LoRA rank | 32 | 16 |
| LoRA alpha | 64 | 32 |
| Learning rate | 1e-4 | 5e-7 |
| Epochs | 3 | 2 |
| Batch size (effective) | 16 | 16 |
| Max sequence length | 4096 | 4096 |
| Precision | bf16 | bf16 |
| Gradient checkpointing | Yes | Yes |
| Num generations (GRPO) | - | 4 |
| KL beta (GRPO) | - | 0.0 |
| Loss computation | assistant_only_loss | GRPO clipped surrogate |

---

## Evaluation Plan

Create `training/eval.py` to evaluate the final model:

```python
# For each difficulty tier (easy, medium, hard) + procedural seeds:
#   1. Run 50 episodes with the trained model
#   2. Record: grader_score, coverage, cost, steps_used
#   3. Compare against: base Qwen2.5-7B, SFT-only model, GRPO model
#
# Metrics to report:
#   - Mean grader score per tier
#   - Pass rate (score >= 0.5) per tier
#   - Mean coverage (passengers booked / total)
#   - Mean cost efficiency
#   - Mean steps used / max steps
#   - Hard constraint violation rate
```

---

## Risk Factors and Mitigations

| Risk | Mitigation |
|------|-----------|
| SFT overfits to expert strategy, doesn't generalize | Use diverse seeds, include suboptimal trajectories, limit epochs |
| GRPO reward is sparse (only at episode end) | Use environment_factory so TRL handles multi-turn; step rewards provide signal during training |
| 7B model can't track 25 passengers | SFT teaches it to use tools (list_passengers) to refresh state; GRPO reinforces this |
| Long episodes exhaust context window | max_length=4096 with LoRA; gradient checkpointing for memory |
| GRPO training is slow (4 generations per prompt) | Use vLLM colocate mode for fast generation; reduce num_generations to 2 if needed |
| Format collapse during GRPO | Low learning rate (5e-7); monitor log_completions for JSON validity |
| environment_factory is experimental in TRL | Have fallback plan using custom reward_func (Section 2.5) |
