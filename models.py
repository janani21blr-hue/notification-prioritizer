from pydantic import BaseModel, Field
from typing import Optional

class NotificationObservation(BaseModel):
    id: int
    app: str
    message: str
    sender: str
    user_state: str
    current_focus: float = Field(..., description="User's remaining attention budget (0.0 to 1.0)")
    is_user_annoyed: bool = Field(..., description="True if focus is below 0.3")

class NotificationAction(BaseModel):
    mode: str = Field(..., pattern="^(notify|delay|ignore)$")

class Reward(BaseModel):
    value: float
    reason: str


