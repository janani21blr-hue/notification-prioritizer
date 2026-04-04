from data import NOTIFICATIONS
from rewards import get_reward

class NotificationEnv:
    def __init__(self, data=None):
        if data is None:
            self.notifications = NOTIFICATIONS
        else:
            self.notifications = data

    def reset(self):
        self.current_index = 0
        self.total_reward = 0
        self.done = False
        return self._get_observation()

    def step(self, action):
        
        if self.done or self.current_index >= len(self.notifications):
            return None, 0.0, True
        notification = self.notifications[self.current_index]
        label = notification["label"]

        reward = get_reward(action.mode, label, notification["user_state"])

        self.total_reward += reward
        self.current_index += 1

        if self.current_index >= len(self.notifications):
            self.done = True
            next_obs = None
        else:
            next_obs = self._get_observation()

        return next_obs, reward, self.done

    def _get_observation(self):
        n = self.notifications[self.current_index]
        return {
            "id":         n["id"],
            "app":        n["app"],
            "message":    n["message"],
            "sender":     n["sender"],
            "user_state": n["user_state"]
        }
