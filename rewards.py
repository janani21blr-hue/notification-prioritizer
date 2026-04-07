REWARD_TABLE = {
    ("notify", "important"): 0.99,
    ("delay", "important"): 0.3,
    ("ignore", "important"): 0.01,

    ("notify", "optional"): 0.7,
    ("delay", "optional"): 0.8,
    ("ignore", "optional"): 0.5,

    ("notify", "ignore"): 0.01,
    ("delay", "ignore"): 0.5,
    ("ignore", "ignore"): 0.99,
}


def get_reward(action_mode, label, user_state):
    # --- Step 1: Base reward ---
    reward = REWARD_TABLE.get((action_mode, label), 0.5)

    # --- Step 2: Adjust based on user state ---
    if user_state == "studying":
        if action_mode == "notify" and label == "ignore":
            reward -= 0.2
        elif action_mode == "ignore" and label == "important":
            reward -= 0.1

    elif user_state == "relaxing":
        if action_mode == "delay" and label == "optional":
            reward += 0.1

    # --- Step 3: FINAL CLAMP (CRITICAL FOR PASSING) ---
    reward = float(reward)
    reward = max(0.01, min(0.99, reward))

    return reward