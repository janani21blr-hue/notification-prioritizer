# train.py
import random
import json
import matplotlib.pyplot as plt
from env import NotificationEnv
from data import NOTIFICATIONS
from tasks import task_mixed
import agent as ag

EPISODES = 500
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99

def train():
    ag.q_table = {}
    epsilon = EPSILON_START
    episode_rewards = []

    for ep in range(EPISODES):
        data = task_mixed(NOTIFICATIONS)
        env = NotificationEnv(data=data)
        obs, _ = env.reset()
        total = 0.0
        done = False

        while not done and obs is not None:
            state_key = ag.get_state_key(obs)
            ag._ensure_state(state_key)

            if random.random() < epsilon:
                action = random.choice(ag.ACTIONS)
            else:
                values = ag.q_table[state_key]
                best_val = max(values.values())
                best_actions = [a for a, v in values.items() if v == best_val]
                action = random.choice(best_actions)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward

            next_key = ag.get_state_key(next_obs) if next_obs is not None else None
            ag.update_q(state_key, action, reward, next_key)
            obs = next_obs

        episode_rewards.append(total)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if (ep + 1) % 50 == 0:
            avg = sum(episode_rewards[-50:]) / 50
            print(f"Ep {ep+1}/500  avg_reward={avg:.3f}  epsilon={epsilon:.3f}")

    ag.save_q_table()

    # Save reward history for dashboard to load
    with open("reward_history.json", "w") as f:
        json.dump(episode_rewards, f)

    print(f"\nDone. Learned {len(ag.q_table)} states.")
    print("Saved q_table.json + reward_history.json")

    # Rolling average curve
    window = 25
    rolling = [
        sum(episode_rewards[max(0, i - window):i + 1]) / min(i + 1, window)
        for i in range(len(episode_rewards))
    ]
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.2, color="#6366f1", label="Per-episode reward")
    plt.plot(rolling, linewidth=2.5, color="#6366f1", label=f"Rolling avg (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Attention Guard: Q-learning Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("learning_curve.png", dpi=120, bbox_inches="tight")
    print("Saved learning_curve.png")

if __name__ == "__main__":
    train()