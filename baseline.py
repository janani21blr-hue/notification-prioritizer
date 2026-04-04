import random
from env import NotificationEnv
from data import NOTIFICATIONS
from agent import smart_agent_action
from tasks import task_urgent,task_mixed,task_noisy
from models import Action


# Fix randomness (important for reproducibility)
random.seed(42)


ACTIONS = ["notify", "ignore", "delay"]


def run_random_agent(data, episodes=20):
    scores = []

    for _ in range(episodes):
        env = NotificationEnv(data=data)
        obs = env.reset()
        done = False

        while not done:
            action = random.choice(ACTIONS)
            action_obj = Action(mode=action, notification_id=obs["id"])
            obs, reward, done = env.step(action_obj)

        scores.append(env.total_reward)

    return sum(scores) / len(scores)


def run_smart_agent(data, episodes=10):
    scores = []

    for _ in range(episodes):
        env = NotificationEnv(data=data)
        obs = env.reset()
        done = False

        while not done:
            action = smart_agent_action(obs)
            action_obj = Action(mode=action, notification_id=obs["id"])
            obs, reward, done = env.step(action_obj)

        scores.append(env.total_reward)

    return sum(scores) / len(scores)


def main():
    tasks = {
        "URGENT": task_urgent(NOTIFICATIONS),
        "MIXED": task_mixed(NOTIFICATIONS),
        "NOISY": task_noisy(NOTIFICATIONS)
    }

    print("\n=== BASELINE EVALUATION ===\n")

    for name, data in tasks.items():
        random_score = run_random_agent(data)
        smart_score = run_smart_agent(data)

        print(f"--- {name} TASK ---")
        print(f"Random Agent: {round(random_score, 2)}")
        print(f"Smart Agent : {round(smart_score, 2)}")

        improvement = ((smart_score - random_score) / max(random_score, 1e-6)) * 100
        print(f"Improvement : {round(improvement, 2)}%\n")


if __name__ == "__main__":
    main()