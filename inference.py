import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the project root (one level up from this file's directory)
load_dotenv(Path(__file__).parent.parent / ".env")

from client import SeatSwapEnv
from models import SeatSwapAction

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
IMAGE_NAME = os.getenv("IMAGE_NAME")
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TASK_NAME = "seat_reassignment"
BENCHMARK = "seat_swap"
MAX_STEPS = 60
TEMPERATURE = 0.3
MAX_TOKENS = 300

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an airline operations agent. An aircraft swap has occurred — all 20 passengers from Aircraft-1 (AC-1) must be reassigned to Aircraft-2 (AC-2). The two aircraft have the same number of seats per cabin class but different seating layouts.

YOUR GOAL:
Reassign every passenger from AC-1 to AC-2 while:
1. Maintaining cabin class (business passengers stay in business, economy in economy)
2. Satisfying paid preferences (passengers who paid for window seats should get window seats on AC-2)

TOOLS AVAILABLE:
You have three tools. Each turn you must call exactly one tool.

1. get_passenger_details(seat_id)
   - Fetches the passenger info for an AC-1 seat
   - Returns: passenger_id, name, cabin, paid_window, current_seat_type
   - Use this BEFORE assigning to learn passenger preferences

2. assign_seat(passenger_id, target_seat_id)
   - Moves a passenger from AC-1 to a specific AC-2 seat
   - The passenger must still be on AC-1, the AC-2 seat must be empty
   - Returns: confirmation with cabin_match and preference_satisfied status

3. swap_seats(passenger_id_1, passenger_id_2)
   - Swaps two passengers who are BOTH already on AC-2
   - Use this to fix earlier mistakes (e.g., a paid-window passenger in a middle seat)

ACTION FORMAT:
Respond with ONLY a JSON object, no other text:
{"tool_name": "get_passenger_details", "args": {"seat_id": "1A"}}
{"tool_name": "assign_seat", "args": {"passenger_id": "PAX-003", "target_seat_id": "3A"}}
{"tool_name": "swap_seats", "args": {"passenger_id_1": "PAX-003", "passenger_idþ_2": "PAX-007"}}

STRATEGY:
1. Start by fetching passenger details for constrained seats (business window seats first, then economy window seats) — these passengers are hardest to place correctly.
2. Assign passengers whose preferences you know, matching cabin class and seat type.
3. After assigning preference-critical passengers, fill remaining seats by cabin class.
4. If you discover a paid-window passenger was placed incorrectly, use swap_seats to fix it.
5. Window seats on AC-1: business 1A, 1D, 2A, 2D. Economy 3A, 3F, 4A, 4F.
6. Window seats on AC-2: business 1A, 1D, 2A, 2D. Economy 3A, 3H, 4A, 4H.

IMPORTANT:
- Never assign a passenger to a seat in the wrong cabin class.
- Always check passenger details before assigning window-seat passengers.
- Each tool call is one step. You have a maximum of 60 steps.
- You can see the current state of both aircraft after each step."""


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------
def format_observation(obs) -> str:
    """Convert environment observation into a readable prompt for the LLM."""
    parts = []

    parts.append(
        f"=== Step {obs.step_count}/{obs.max_steps} | "
        f"Passengers remaining: {obs.passengers_remaining}/{obs.passengers_total} ==="
    )

    if obs.tool_result is not None:
        parts.append(f"\nLast tool result: {json.dumps(obs.tool_result, indent=2)}")

    if obs.reward is not None and obs.step_count > 0:
        parts.append(f"Reward: {obs.reward:.2f} ({obs.reward_reason})")

    parts.append(f"\nAC-1 seats still occupied (passengers to reassign): {obs.ac1_seats_occupied}")

    if obs.ac2_seats_occupied:
        parts.append(f"\nAC-2 current assignments:")
        for seat_id, pax_id in sorted(obs.ac2_seats_occupied.items()):
            parts.append(f"  {seat_id} -> {pax_id}")
    else:
        parts.append(f"\nAC-2: No passengers assigned yet.")

    parts.append(f"\nAC-2 available seats: {obs.ac2_seats_available}")

    if obs.step_count == 0:
        parts.append(f"\nAC-1 layout: {json.dumps(obs.ac1_layout['layout'], indent=2)}")
        parts.append(f"AC-2 layout: {json.dumps(obs.ac2_layout['layout'], indent=2)}")

    parts.append("\nChoose your next tool call. Respond with ONLY a JSON object.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------
def parse_llm_response(response_text: str) -> Optional[dict]:
    """Parse LLM response into a tool call dict. Returns None on failure."""
    text = response_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if "tool_name" in parsed and "args" in parsed:
            return parsed
        return None
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"tool_name"[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def get_agent_action(client: OpenAI, obs, conversation_history: list) -> Optional[dict]:
    """Call the LLM to get the next action."""
    user_prompt = format_observation(obs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for entry in conversation_history[-6:]:
        messages.append({"role": "assistant", "content": entry["action"]})
        messages.append({"role": "user", "content": entry["observation"]})

    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        parsed = parse_llm_response(response_text)

        if parsed is None:
            print(f"[DEBUG] Failed to parse LLM response: {response_text[:200]}", flush=True)
            return None

        return parsed
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Fallback action
# ---------------------------------------------------------------------------
def fallback_action(obs) -> dict:
    """Simple fallback: fetch the first unqueried AC-1 seat."""
    if obs.ac1_seats_occupied and obs.ac2_seats_available:
        seat_id = obs.ac1_seats_occupied[0]
        return {"tool_name": "get_passenger_details", "args": {"seat_id": seat_id}}
    return {"tool_name": "get_passenger_details", "args": {"seat_id": "1A"}}


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
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = SeatSwapEnv(base_url="http://localhost:8000")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    conversation_history: list = []

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        print(f"\n{'='*60}", flush=True)
        print(f"Episode started: {obs.passengers_total} passengers to reassign", flush=True)
        print(f"AC-1 seats occupied: {len(obs.ac1_seats_occupied)}", flush=True)
        print(f"AC-2 seats available: {len(obs.ac2_seats_available)}", flush=True)
        print(f"Max steps: {obs.max_steps}", flush=True)
        print(f"{'='*60}\n", flush=True)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_dict = get_agent_action(client, obs, conversation_history)

            if action_dict is None:
                print(f"[DEBUG] Step {step}: LLM failed, using fallback", flush=True)
                action_dict = fallback_action(obs)

            action_summary = f"{action_dict['tool_name']}({json.dumps(action_dict['args'])})"
            print(f"\n--- Step {step} ---", flush=True)
            print(f"  Action: {action_summary}", flush=True)

            result = await env.step(SeatSwapAction(
                tool_name=action_dict["tool_name"],
                args=action_dict["args"],
            ))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            print(f"  Reward: {reward:.2f} ({obs.reward_reason})", flush=True)
            print(f"  Passengers remaining: {obs.passengers_remaining}/{obs.passengers_total}", flush=True)
            if obs.tool_result:
                status = obs.tool_result.get("status", "unknown")
                print(f"  Tool status: {status}", flush=True)
                if status == "error":
                    print(f"  Error: {obs.tool_result.get('message', 'unknown')}", flush=True)

            rewards.append(reward)
            steps_taken = step

            error = (
                obs.tool_result.get("message")
                if obs.tool_result and obs.tool_result.get("status") == "error"
                else None
            )
            log_step(step=step, action=action_summary, reward=reward, done=done, error=error)

            conversation_history.append({
                "action": json.dumps(action_dict),
                "observation": format_observation(obs),
            })

            if done:
                print(f"\n{'='*60}", flush=True)
                print(f"Episode complete at step {step}!", flush=True)
                if obs.tool_result and "grader_score" in obs.tool_result:
                    score = obs.tool_result["grader_score"]
                    print(f"Grader score: {score:.3f}", flush=True)
                print(f"{'='*60}\n", flush=True)
                break

        success = score >= 0.5
        score = min(max(score, 0.0), 1.0)

    except Exception as e:
        print(f"[DEBUG] Episode failed with exception: {e}", flush=True)
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
