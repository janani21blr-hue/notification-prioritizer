# baseline.py
import random
from env import NotificationEnv

def run_random_agent(data):
    env = NotificationEnv(data=data)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    actions_taken = {"notify": 0, "delay": 0, "ignore": 0}
    focus_history = []

    while not done and obs is not None:
        action = random.choice(["notify", "delay", "ignore"])
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        actions_taken[action] += 1
        focus_history.append(info["current_focus"])
        obs = next_obs
        done = terminated or truncated

    return {
        "total_reward": round(total_reward, 3),
        "avg_reward": round(total_reward / max(len(focus_history), 1), 3),
        "final_focus": round(focus_history[-1] if focus_history else 0.0, 2),
        "avg_focus": round(sum(focus_history) / max(len(focus_history), 1), 3),
        "actions": actions_taken,
        "steps": len(focus_history),
    }

def run_trained_agent(data):
    from agent import choose_action
    env = NotificationEnv(data=data)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    actions_taken = {"notify": 0, "delay": 0, "ignore": 0}
    focus_history = []

    while not done and obs is not None:
        action = choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        actions_taken[action] += 1
        focus_history.append(info["current_focus"])
        obs = next_obs
        done = terminated or truncated

    return {
        "total_reward": round(total_reward, 3),
        "avg_reward": round(total_reward / max(len(focus_history), 1), 3),
        "final_focus": round(focus_history[-1] if focus_history else 0.0, 2),
        "avg_focus": round(sum(focus_history) / max(len(focus_history), 1), 3),
        "actions": actions_taken,
        "steps": len(focus_history),
    }