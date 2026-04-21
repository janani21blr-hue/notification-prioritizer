# env.py
# env.py
from data import NOTIFICATIONS
from rewards import get_reward
from models import NotificationObservation, NotificationAction

class NotificationEnv:
    def __init__(self, data=None):
        # Use provided task data or default to the full list
        self.notifications = data if data else NOTIFICATIONS
        self.reset()

    def reset(self):
        """Resets the environment to the starting state."""
        self.current_index = 0
        self.total_reward = 0.0
        self.focus_level = 1.0  # User starts at 100% focus
        self.done = False
        return self._get_observation()

    def step(self, action):
        """
        Executes one action and returns (observation, reward, done).
        'action' can be a string ('notify') or a NotificationAction object.
        """
        if self.done or self.current_index >= len(self.notifications):
            return None, 0.0, True

        # Get current notification details
        notif = self.notifications[self.current_index]
        
        # Handle the action object or raw string
        mode = action.mode if hasattr(action, "mode") else str(action)

        # 1. Calculate Reward based on CURRENT focus before it changes
        reward = get_reward(
            mode, 
            notif.get("label", "optional"), 
            notif["user_state"], 
            self.focus_level
        )

        # 2. State Transition: How does this action change the user's world?
        if mode == "notify":
            self.focus_level -= 0.20  # Interruption cost
        else:
            self.focus_level += 0.05  # Silence allows recovery
        
        # Keep focus level within logical bounds [0.0, 1.0]
        self.focus_level = max(0.0, min(1.0, self.focus_level))

        self.total_reward += reward
        self.current_index += 1

        # Check if episode is finished
        if self.current_index >= len(self.notifications):
            self.done = True
            return None, float(reward), True

        return self._get_observation(), float(reward), self.done

    def _get_observation(self) -> NotificationObservation:
        """Helper to package the current state into the standard model."""
        if self.current_index >= len(self.notifications):
            return None
            
        n = self.notifications[self.current_index]
        
        # We map the raw dictionary to our Pydantic model
        return NotificationObservation(
            id=n["id"],
            app=n["app"],
            message=n["message"],
            sender=n["sender"],
            user_state=n["user_state"],
            current_focus=round(self.focus_level, 2),
            is_user_annoyed=self.focus_level < 0.3
        )