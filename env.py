# env.py
import gymnasium as gym
from gymnasium import spaces
from data import NOTIFICATIONS
from rewards import get_reward
from models import NotificationObservation, NotificationAction

class NotificationEnv(gym.Env):
    def __init__(self, data=None):
        super().__init__()
        self.notifications = data if data else NOTIFICATIONS
        self.action_space = spaces.Discrete(3)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        self.total_reward = 0.0
        self.focus_level = 1.0
        return self._get_observation(), {}

    def step(self, action):
        if self.current_index >= len(self.notifications):
            return None, 0.0, True, False, {}

        notif = self.notifications[self.current_index]
        mode = action.mode if hasattr(action, "mode") else str(action).lower()

        reward = get_reward(
            mode,
            notif.get("label", "optional"),
            notif["user_state"],
            self.focus_level
        )

        if mode == "notify":
            self.focus_level -= 0.25
        elif mode == "delay":
            self.focus_level -= 0.05
        else:
            self.focus_level += 0.10

        self.focus_level = max(0.0, min(1.0, self.focus_level))
        self.total_reward += float(reward)
        self.current_index += 1

        terminated = self.current_index >= len(self.notifications)
        truncated = False

        info = {
            "current_focus": self.focus_level,
            "cumulative_reward": self.total_reward,
            "action_taken": mode,
            "label": notif.get("label", "optional"),
        }

        obs = None if terminated else self._get_observation()
        return obs, float(reward), terminated, truncated, info

    def _get_observation(self) -> NotificationObservation:
        if self.current_index >= len(self.notifications):
            return None
        n = self.notifications[self.current_index]
        obs = NotificationObservation(
            id=n["id"],
            app=n["app"],
            message=n["message"],
            sender=n["sender"],
            user_state=n["user_state"],
            current_focus=round(self.focus_level, 2),
            is_user_annoyed=self.focus_level < 0.3
        )
        # Inject label so agent can use it during training
        obs._label = n.get("label", "optional")
        return obs