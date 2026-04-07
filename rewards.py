# rewards.py

REWARD_TABLE = {
    ("notify", "important"): 0.95,
    ("delay",  "important"): 0.30,
    ("ignore", "important"): 0.05,

    ("notify", "optional"):  0.70,
    ("delay",  "optional"):  0.85,
    ("ignore", "optional"):  0.40,

    ("notify", "junk"):      0.05,
    ("delay",  "junk"):      0.50,
    ("ignore", "junk"):      0.95,
}

def get_reward(action_mode, label, user_state):
    # Ensure label matches the table (maps 'ignore' to 'junk' if needed)
    clean_label = "junk" if label in ["junk", "ignore", "low"] else label
    
    # Base reward with a safe 0.5 fallback
    reward = REWARD_TABLE.get((action_mode, clean_label), 0.5)

    # Adjustments
    if user_state == "studying":
        if action_mode == "notify" and clean_label == "junk":
            reward -= 0.15
        elif action_mode == "ignore" and clean_label == "important":
            reward -= 0.10
    elif user_state == "relaxing":
        if action_mode == "delay" and clean_label == "optional":
            reward += 0.05

    # CRITICAL: Tight boundary enforcement
    # This keeps rewards strictly in (0.01, 0.99)
    return float(max(0.01, min(0.99, reward)))