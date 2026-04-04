REWARD_TABLE = {
    ("notify", "important"): 1.0,
    ("delay",  "important"): 0.3,
    ("ignore", "important"): 0.0,

    ("notify", "optional"):  0.7,
    ("delay",  "optional"):  0.8,
    ("ignore", "optional"):  0.5,

    ("notify", "ignore"):    0.0,
    ("delay",  "ignore"):    0.5,
    ("ignore", "ignore"):    1.0,
}

def get_reward(action_mode, label, user_state):
    reward = REWARD_TABLE[(action_mode, label)]

    if user_state == "studying":
        if action_mode == "notify" and label == "ignore":
            reward = max(0.0, reward - 0.2)
        elif action_mode == "ignore" and label == "important":
            reward = max(0.0, reward - 0.1)
    elif user_state == "relaxing":
        if action_mode == "delay" and label == "optional":
            reward = min(1.0, reward + 0.1)

    return reward