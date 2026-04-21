# baseline.py
# baseline.py
import random
from env import NotificationEnv
from data import NOTIFICATIONS
from agent import agent_step

def run_benchmark(task_name, data):
    env = NotificationEnv(data=data)
    obs = env.reset()
    done = False
    
    print(f"\n--- Benchmark: {task_name} ---")
    while not done:
        # Pass the observation object to the agent
        action_str, _ = agent_step(obs)
        obs, reward, done = env.step(action_str)
        print(f"Focus: {obs.current_focus if obs else '0'} | Action: {action_str} | Reward: {reward}")
    
    print(f"Total Reward for {task_name}: {round(env.total_reward, 2)}")

if __name__ == "__main__":
    from tasks import task_mixed
    run_benchmark("MIXED_TASK", task_mixed(NOTIFICATIONS))