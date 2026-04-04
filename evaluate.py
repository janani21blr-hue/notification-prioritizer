from env import NotificationEnv
from agent import smart_agent_action
from models import Action
import random

def random_agent_action(obs=None):
    return random.choice(["notify", "delay", "ignore"])

def run_agent(agent_fn, name):
    env = NotificationEnv()
    obs = env.reset()
    done = False

    while not done:
        if obs is None:
            break
        action = agent_fn(obs) 
        action_obj = Action(mode=action)

        obs, reward, done = env.step(action_obj)

    print(f"{name} Total reward: {round(env.total_reward, 2)}")


if __name__ == "__main__":
    run_agent(random_agent_action, "Random Agent")
    run_agent(smart_agent_action, "Smart Agent")