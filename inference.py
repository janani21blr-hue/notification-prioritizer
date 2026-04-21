# inference.py
import os
import time
import requests
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")

VALID_ACTIONS = {"notify", "delay", "ignore"}

SYSTEM_PROMPT = """You are an AI Attention Guard. 
Decide if a notification should be shown NOW, DELAYED, or IGNORED.
Consider the User's Focus Budget and Annoyance state.
Output ONLY the word: notify, delay, or ignore."""

# --- LLM LOGIC ---
def get_llm_action(client: OpenAI, obs: dict) -> str:
    try:
        focus = obs.get("current_focus", 1.0)
        annoyed = obs.get("is_user_annoyed", False)
        
        prompt = (
            f"User Focus: {int(focus*100)}% | Annoyed: {annoyed}\n"
            f"App: {obs.get('app')} | Message: {obs.get('message')}\n"
            "Action:"
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        
        raw = response.choices[0].message.content.strip().lower()
        for action in VALID_ACTIONS:
            if action in raw: return action
        return "ignore"
    except Exception as e:
        print(f"LLM Error: {e}")
        return "ignore"

# --- RUNNER ---
def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in ["urgent", "mixed", "noisy"]:
        print(f"\n--- Starting Task: {task} ---")
        try:
            r = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task})
            res = r.json()
            obs = res.get("observation")
            done = False
            
            while not done:
                action = get_llm_action(client, obs)
                # CRITICAL: We send 'mode' to match our models.py
                step_r = requests.post(f"{ENV_BASE_URL}/step", json={"action": action})
                step_res = step_r.json()
                
                obs = step_res.get("observation")
                reward = step_res.get("reward")
                done = step_res.get("done")
                print(f"Action: {action} | Focus: {obs['current_focus'] if obs else 'End'} | Reward: {reward}")
        except Exception as e:
            print(f"Connection Error: {e}")

if __name__ == "__main__":
    main()