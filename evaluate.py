# evaluate.py
# Runs one episode per agent on the default task and prints totals.
# Intended as a quick sanity-check; baseline.py does the multi-episode average.

import random
from env import NotificationEnv
from agent import smart_agent_action
from models import Action


def random_agent_action(obs=None):
    return random.choice(["notify", "delay", "ignore"])


def run_agent(agent_fn, name):
    env = NotificationEnv()
    obs = env.reset()
    done = False

    while not done and obs is not None:
        action = agent_fn(obs)
        action_obj = Action(mode=action)
        obs, _, done = env.step(action_obj)

    print(f"{name:15s} total reward: {round(env.total_reward, 2)}")


if __name__ == "__main__":
    run_agent(random_agent_action, "Random Agent")
    run_agent(smart_agent_action, "Smart Agent")