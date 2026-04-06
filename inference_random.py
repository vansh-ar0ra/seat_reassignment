import asyncio
import json
import os
import random
from typing import List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "airline_reassignment_inference"))
from client import AirlineReassignmentEnv
from models import AirlineReassignmentAction

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("IMAGE_NAME")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TASK_NAME = "seat_reassignment_random"
BENCHMARK = "airline_reassignment"
MAX_STEPS = 60
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Mandatory logging
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def main() -> None:
    random.seed(RANDOM_SEED)

    env = AirlineReassignmentEnv(base_url="http://localhost:8000")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Track fetched passenger info: seat_id -> passenger details dict
    known_passengers: dict = {}

    log_start(task=TASK_NAME, env=BENCHMARK, model="random_baseline")

    try:
        result = await env.reset()
        obs = result.observation

        print(f"\nRandom baseline: {obs.passengers_total} passengers to reassign", flush=True)
        print(f"AC-2 available seats: {len(obs.ac2_seats_available)}", flush=True)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            unfetched_seats = [s for s in obs.ac1_seats_occupied if s not in known_passengers]
            unassigned_known = {
                seat: info
                for seat, info in known_passengers.items()
                if seat in obs.ac1_seats_occupied
            }

            if unfetched_seats and (not unassigned_known or random.random() < 0.3):
                seat_id = random.choice(unfetched_seats)
                action_dict = {"tool_name": "get_passenger_details", "args": {"seat_id": seat_id}}
            elif unassigned_known and obs.ac2_seats_available:
                seat_id = random.choice(list(unassigned_known.keys()))
                pax_info = unassigned_known[seat_id]
                target_seat = random.choice(obs.ac2_seats_available)
                action_dict = {
                    "tool_name": "assign_seat",
                    "args": {
                        "passenger_id": pax_info["passenger_id"],
                        "target_seat_id": target_seat,
                    },
                }
            elif unfetched_seats:
                seat_id = random.choice(unfetched_seats)
                action_dict = {"tool_name": "get_passenger_details", "args": {"seat_id": seat_id}}
            else:
                print(f"[DEBUG] Step {step}: No actions available, breaking", flush=True)
                break

            action_summary = f"{action_dict['tool_name']}({json.dumps(action_dict['args'])})"
            print(f"Step {step}: {action_summary}", flush=True)

            result = await env.step(AirlineReassignmentAction(
                tool_name=action_dict["tool_name"],
                args=action_dict["args"],
            ))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            if (
                action_dict["tool_name"] == "get_passenger_details"
                and obs.tool_result
                and obs.tool_result.get("status") == "success"
            ):
                fetched_seat = action_dict["args"]["seat_id"]
                known_passengers[fetched_seat] = obs.tool_result

            print(f"  Reward: {reward:.2f} | Remaining: {obs.passengers_remaining}/{obs.passengers_total}", flush=True)

            rewards.append(reward)
            steps_taken = step

            error = (
                obs.tool_result.get("message")
                if obs.tool_result and obs.tool_result.get("status") == "error"
                else None
            )
            log_step(step=step, action=action_summary, reward=reward, done=done, error=error)

            if done:
                if obs.tool_result and "grader_score" in obs.tool_result:
                    score = obs.tool_result["grader_score"]
                print(f"\nRandom baseline complete. Grader score: {score:.3f}", flush=True)
                break

        success = score >= 0.5
        score = min(max(score, 0.0), 1.0)

    except Exception as e:
        print(f"[DEBUG] Episode failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
