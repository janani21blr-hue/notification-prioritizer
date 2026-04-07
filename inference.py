import os
import requests
from typing import List, Optional
from openai import OpenAI

# --- ENV VARIABLES ---
API_KEY = os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://digitalpixie-notification-prioritizer.hf.space")
TASK_NAME = os.getenv("TASK", "mixed")
BENCHMARK = "notification-prioritizer"

MAX_STEPS = 20

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

# --- LOGGING (STRICT FORMAT) ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# --- LLM DECISION ---
def get_llm_action(client: OpenAI, obs):
    prompt = f"""
App: {obs['app']}
Message: {obs['message']}
Sender: {obs['sender']}
User state: {obs['user_state']}

Choose one action: notify, delay, or ignore.
"""

    try:
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

        if "notify" in raw:
            return "notify"
        elif "delay" in raw:
            return "delay"
        elif "ignore" in raw:
            return "ignore"
        else:
            return "ignore"

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "ignore"

# --- MAIN LOOP ---
def main():
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )
    rewards = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task": TASK_NAME})
        result = reset_resp.json()
        obs = result["observation"]
        done = result["done"]

        step = 0

        while not done and step < MAX_STEPS:
            step += 1

            action = get_llm_action(client, obs)
            action = action.strip().lower()

            if action not in ["notify", "delay", "ignore"]:
                action = "ignore"

            step_resp = requests.post(
                f"{ENV_BASE_URL}/step",
                json={"action": action}
            )
            step_result = step_resp.json()

            obs = step_result["observation"]
            reward = step_result["reward"]
            done = step_result["done"]
            error = step_result.get("error")

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, error)

        success = sum(rewards) >= 1

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()