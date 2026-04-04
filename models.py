from pydantic import BaseModel
from typing import Literal

class Action(BaseModel):
    mode: Literal["notify", "delay", "ignore"]
    

class Observation(BaseModel):
    id: int
    app: str
    message: str
    sender: str
    user_state: Literal["studying", "relaxing"]

class State(BaseModel):
    current_index: int
    total_reward: float
    done: bool


