from typing import Optional
from pydantic import BaseModel
from typing import Literal

class Reward(BaseModel):
    value: float

class Action(BaseModel):
    mode: Literal["notify", "delay", "ignore"]
    notification_id: Optional[int] = None  # add this
    

class Observation(BaseModel):
    id: int
    app: str
    message: str
    sender: str
    user_state: str

class State(BaseModel):
    current_index: int
    total_reward: float
    done: bool


