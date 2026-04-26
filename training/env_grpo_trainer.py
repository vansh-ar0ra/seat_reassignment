"""GRPOTrainer subclass that supports environment-based rollouts on TRL 0.19.

TRL 0.19's GRPOTrainer has no ``rollout_func`` or ``env`` parameter (those
were added in TRL 0.26).  This subclass overrides
``_generate_and_score_completions`` to drive multi-turn episodes via the
rollout function, then converts the results into the tensor format that
TRL 0.19's ``_compute_loss`` expects.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
from trl import GRPOTrainer

logger = logging.getLogger("flight_rebooking.env_grpo_trainer")


class EnvGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with environment rollout support for TRL 0.19."""

    def __init__(
        self,
        *args: Any,
        rollout_func: Optional[Callable] = None,
        **kwargs: Any,
    ):
        self._rollout_func = rollout_func
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Override: replace standard generation with environment rollouts
    # ------------------------------------------------------------------
    def _generate_and_score_completions(
        self, inputs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Run environment episodes instead of standard model generation.

        The rollout function returns per-episode:
            prompt_ids, completion_ids, logprobs, env_mask,
            + reward component floats

        We convert these into the padded-tensor format that
        ``_compute_loss`` expects:
            prompt_ids, prompt_mask, completion_ids, completion_mask,
            advantages, old_per_token_logps, ref_per_token_logps
        """
        if self._rollout_func is None:
            return super()._generate_and_score_completions(inputs)

        device = self.accelerator.device

        # --- Run rollout ---
        prompts = inputs["prompt"]
        rollout = self._rollout_func(prompts, self)

        raw_prompt_ids = rollout["prompt_ids"]       # list[list[int]]
        raw_completion_ids = rollout["completion_ids"]  # list[list[int]]
        raw_logprobs = rollout["logprobs"]           # list[list[float]]
        raw_env_mask = rollout.get("env_mask")       # list[list[int]] | None

        batch_size = len(raw_prompt_ids)

        # --- Pad prompt_ids ---
        max_prompt_len = max(len(p) for p in raw_prompt_ids)
        pad_id = self.processing_class.pad_token_id or 0
        prompt_ids = torch.full(
            (batch_size, max_prompt_len), pad_id, dtype=torch.long, device=device
        )
        prompt_mask = torch.zeros(
            (batch_size, max_prompt_len), dtype=torch.long, device=device
        )
        for i, pids in enumerate(raw_prompt_ids):
            L = len(pids)
            # Right-align prompts (left-pad)
            prompt_ids[i, max_prompt_len - L :] = torch.tensor(pids, dtype=torch.long)
            prompt_mask[i, max_prompt_len - L :] = 1

        # --- Pad completion_ids ---
        max_comp_len = max(len(c) for c in raw_completion_ids) if raw_completion_ids else 1
        completion_ids = torch.full(
            (batch_size, max_comp_len), pad_id, dtype=torch.long, device=device
        )
        completion_mask = torch.zeros(
            (batch_size, max_comp_len), dtype=torch.long, device=device
        )
        for i, cids in enumerate(raw_completion_ids):
            L = len(cids)
            completion_ids[i, :L] = torch.tensor(cids, dtype=torch.long)
            if raw_env_mask is not None:
                # Use env_mask: 1 = model token (train on), 0 = env token (mask out)
                completion_mask[i, :L] = torch.tensor(
                    raw_env_mask[i][:L], dtype=torch.long
                )
            else:
                completion_mask[i, :L] = 1

        # --- Build old_per_token_logps from rollout logprobs ---
        old_per_token_logps = torch.zeros(
            (batch_size, max_comp_len), dtype=torch.float32, device=device
        )
        for i, lps in enumerate(raw_logprobs):
            L = len(lps)
            old_per_token_logps[i, :L] = torch.tensor(lps, dtype=torch.float32)

        # --- Compute rewards from reward_funcs ---
        # Decode completions for reward functions
        completions_text = []
        for cids in raw_completion_ids:
            completions_text.append(
                self.processing_class.decode(cids, skip_special_tokens=True)
            )

        # Build reward kwargs from rollout extras (reward component scores)
        reward_kwargs: dict[str, Any] = {}
        for key in rollout:
            if key not in ("prompt_ids", "completion_ids", "logprobs", "env_mask"):
                reward_kwargs[key] = rollout[key]

        # Call each reward function
        rewards_per_func = []
        for rf in self.reward_funcs:
            if callable(rf):
                scores = rf(
                    prompts=prompts * getattr(self.args, "num_generations", 1)
                    if len(prompts) != batch_size
                    else prompts,
                    completions=completions_text,
                    **reward_kwargs,
                )
                rewards_per_func.append(scores)
            else:
                rewards_per_func.append([0.0] * batch_size)

        # Shape: (batch_size, num_reward_funcs)
        rewards_tensor = torch.tensor(
            list(zip(*rewards_per_func)), dtype=torch.float32, device=device
        )

        # Weighted sum
        reward_weights = self.reward_weights.to(device)
        rewards = (rewards_tensor * reward_weights.unsqueeze(0)).sum(dim=1)

        # --- Group-relative advantages ---
        num_generations = getattr(self.args, "num_generations", 1)
        if num_generations > 1 and batch_size >= num_generations:
            grouped = rewards.view(-1, num_generations)
            mean_g = grouped.mean(dim=1, keepdim=True)
            std_g = grouped.std(dim=1, keepdim=True)
            advantages = (rewards - mean_g.repeat(1, num_generations).view(-1))
            std_flat = std_g.repeat(1, num_generations).view(-1)
            advantages = advantages / (std_flat + 1e-4)
        else:
            advantages = rewards - rewards.mean()
            std_r = rewards.std()
            if std_r > 1e-4:
                advantages = advantages / std_r

        # --- ref_per_token_logps: set to None or zeros ---
        # When beta=0 or using PEFT, TRL 0.19 skips KL penalty
        ref_per_token_logps = None
        if getattr(self, "beta", 0.0) > 0 and not hasattr(self.model, "peft_config"):
            ref_per_token_logps = torch.zeros_like(old_per_token_logps)

        # --- Log metrics ---
        self._metrics["rewards/mean"] = rewards.mean().item()
        self._metrics["rewards/std"] = rewards.std().item()
        self._metrics["completion_length"] = float(
            completion_mask.sum(dim=1).float().mean().item()
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }
