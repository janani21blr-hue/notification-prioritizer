# rewards.py

REWARD_TABLE = {
    ("notify", "important"): 0.95, ("delay", "important"): 0.30, ("ignore", "important"): 0.05,
    ("notify", "optional"): 0.70, ("delay", "optional"): 0.85, ("ignore", "optional"): 0.40,
    ("notify", "junk"): 0.05, ("delay", "junk"): 0.50, ("ignore", "junk"): 0.95,
}

def get_reward(action_mode, label, user_state, focus_level):
    clean_label = "junk" if label in ["junk", "ignore", "low"] else label
    
    # 1. Base Reward lookup
    reward = REWARD_TABLE.get((action_mode, clean_label), 0.5)

    # 2. Focus Penalty (The "Innovation" logic)
    if focus_level < 0.3 and action_mode == "notify":
        reward -= 0.4 
    
    # 3. Contextual Adjustment
    if user_state == "studying" and action_mode == "notify" and clean_label != "important":
        reward -= 0.2

    # 4. OpenEnv Boundary Enforcement (Must be strictly between 0 and 1)
    return float(max(0.01, min(0.99, reward)))