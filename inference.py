import os
import requests
from typing import List, Optional
from openai import OpenAI

# --- ENV VARIABLES ---
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://digitalpixie-notification-prioritizer.hf.space").rstrip("/")
BENCHMARK = "notification-prioritizer"

MAX_STEPS = 18
TASKS = ["urgent", "mixed", "noisy"]

SYSTEM_PROMPT = """
You are a notification prioritization agent.

Given a notification, you must choose exactly one action:
- notify
- delay
- ignore

IMPORTANT:
- Output ONLY one word
- No punctuation
- No explanation
- No extra text

Valid outputs:
notify
delay
ignore
"""

VALID_ACTIONS = {"notify", "delay", "ignore"}

# --- LOGGING ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(str(r) for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# --- SAFE REWARD ---
def safe_reward(value) -> float:
    try:
        r = float(value)
    except (TypeError, ValueError):
        r = 0.5

    if r <= 0:
        r = 0.01
    elif r >= 1:
        r = 0.99

    return min(0.99, max(0.01, r))

# --- SAFE JSON ---
def safe_json(response) -> dict:
    try:
        return response.json()
    except Exception:
        return {}

# --- LLM DECISION ---
def get_llm_action(client: OpenAI, obs: dict) -> str:
    try:
        app        = obs.get("app", "unknown")
        message    = obs.get("message", "")
        sender     = obs.get("sender", "unknown")
        user_state = obs.get("user_state", "idle")

        prompt = f"""App: {app}
Message: {message}
Sender: {sender}
User state: {user_state}

Choose one action: notify, delay, or ignore."""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.0
        )

        raw = response.choices[0].message.content.strip().lower()

        for action in VALID_ACTIONS:
            if action in raw:
                return action

        return "ignore"

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "ignore"

# --- MAIN LOOP ---
def main():
    client = None

    # Only initialize if env vars exist (IMPORTANT for proxy)
    if API_BASE_URL and API_KEY:
        try:
            client = OpenAI(
                base_url=API_BASE_URL,
                api_key=API_KEY
            )
        except Exception as e:
            print(f"[DEBUG] Client init error: {e}", flush=True)

    for task in TASKS:
        rewards: List[float] = []
        steps_taken = 0
        success = False

        log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

        try:
            reset_resp = requests.post(
                f"{ENV_BASE_URL}/reset",
                json={"task": task},
                timeout=30
            )
            result = safe_json(reset_resp)

            obs  = result.get("observation") or {}
            done = result.get("done", False)

            if not obs:
                log_end(False, 0, [])
                continue

            step = 0
            while not done and step < MAX_STEPS:
                step += 1

                if client:
                    action = get_llm_action(client, obs)
                else:
                    action = "ignore"

                if action not in VALID_ACTIONS:
                    action = "ignore"

                step_resp = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json={"action": action},
                    timeout=30
                )
                step_result = safe_json(step_resp)

                obs    = step_result.get("observation") or {}
                reward = safe_reward(step_result.get("reward", 0.5))
                done   = step_result.get("done", True)
                error  = step_result.get("error")

                rewards.append(reward)
                steps_taken = step

                log_step(step, action, reward, done, error)

            success = sum(rewards) > 0.99

        except Exception as e:
            print(f"[DEBUG] Runtime error: {e}", flush=True)

        finally:
            log_end(success, steps_taken, rewards)

if __name__ == "__main__":
    main()