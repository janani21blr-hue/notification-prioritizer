# env.py
from data import NOTIFICATIONS
from rewards import get_reward

class NotificationEnv:
    def __init__(self, data=None):
        self.notifications = data if data else NOTIFICATIONS
        self.action_map = {0: "notify", 1: "delay", 2: "ignore"}
        self.reset()

    def reset(self):
        self.current_index = 0
        self.total_reward = 0.0
        self.done = False
        return self._get_observation()

    def step(self, action):
        # If the validator accidentally over-steps, return a safe middle-ground reward
        if self.done or self.current_index >= len(self.notifications):
            return None, 0.5, True

        notification = self.notifications[self.current_index]
        
        # Handle string actions, integer actions, or Pydantic objects
        if hasattr(action, "mode"):
            mode = action.mode
        elif isinstance(action, int):
            mode = self.action_map.get(action, "ignore")
        else:
            mode = str(action)

        reward = get_reward(mode, notification.get("label", "optional"), notification["user_state"])

        self.total_reward += reward
        self.current_index += 1

        if self.current_index >= len(self.notifications):
            self.done = True
            next_obs = None
        else:
            next_obs = self._get_observation()

        return next_obs, float(reward), self.done

    def _get_observation(self):
        if self.current_index >= len(self.notifications):
            return None
        n = self.notifications[self.current_index]
        return {
            "id": n["id"],
            "app": n["app"],
            "message": n["message"],
            "sender": n["sender"],
            "user_state": n["user_state"]
        }