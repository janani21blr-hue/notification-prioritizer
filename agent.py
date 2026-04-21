# agent.py
import random
import json
import os
from models import NotificationAction

# --- CONFIG ---
q_table = {}
Q_TABLE_PATH = "q_table.json"
ACTIONS = ["notify", "delay", "ignore"]
ALPHA = 0.2
EPSILON = 0.05 # Exploration rate

def load_q_table():
    global q_table
    if os.path.exists(Q_TABLE_PATH):
        try:
            with open(Q_TABLE_PATH, "r") as f:
                q_table = json.load(f)
        except:
            q_table = {}

load_q_table()

# --- HEURISTICS ---

def get_importance(obs):
    """Simple heuristic to classify message priority."""
    msg = obs.message.lower()
    # Critical: OTPs, Banks, Emergencies
    if any(k in msg for k in ["otp", "bank", "emergency", "security", "alert"]):
        return "critical"
    # High: Deadlines, Exams, Results
    if any(k in msg for k in ["deadline", "exam", "assignment", "shortlisted", "viva"]):
        return "high"
    # Low: Promotional, Social
    if any(k in msg for k in ["offer", "sale", "discount", "liked", "posted"]):
        return "low"
    return "medium"

def get_state_key(obs):
    """
    Creates a unique string for the Q-table.
    Innovation: We now include the 'Annoyed' state in the memory!
    """
    importance = get_importance(obs)
    annoyance_prefix = "ANNOYED" if obs.is_user_annoyed else "CALM"
    return f"{annoyance_prefix}|{importance}|{obs.user_state}"

# --- DECISION LOGIC ---

def choose_action(obs):
    state_key = get_state_key(obs)
    importance = get_importance(obs)

    # 1. Check Q-Table for experience
    if state_key in q_table:
        values = q_table[state_key]
        if max(values.values()) > 0.1 and random.random() > EPSILON:
            return max(values, key=values.get)

    # 2. Focus-Aware Fallback Logic
    # If the user is annoyed, be EXTREMELY strict.
    if obs.is_user_annoyed:
        if importance == "critical":
            return "notify"
        return "delay" # Force-delay medium/high priority to save user focus

    # Standard Fallback
    if importance in ["critical", "high"]: return "notify"
    if importance == "medium": return "delay"
    return "ignore"

# --- PUBLIC API ---

def agent_step(obs):
    """
    Main entry point for the app and baseline scripts.
    Returns (action_string, dummy_prior)
    """
    action = choose_action(obs)
    # We return 0.9 as a dummy prior to keep your app.py logic happy
    return action, 0.9

def smart_agent_action(obs) -> NotificationAction:
    """Returns the action wrapped in the Pydantic model for the server."""
    mode = choose_action(obs)
    return NotificationAction(mode=mode)