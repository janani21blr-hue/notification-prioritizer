from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from env import NotificationEnv
from data import NOTIFICATIONS
from models import Action
from tasks import task_urgent, task_mixed, task_noisy


app = FastAPI()

env_instance: Optional[NotificationEnv] = None

TASKS = {
    "urgent": task_urgent(NOTIFICATIONS),
    "mixed": task_mixed(NOTIFICATIONS),
    "noisy": task_noisy(NOTIFICATIONS),
}

class ResetRequest(BaseModel):
    task: Optional[str] = "mixed"

class StepRequest(BaseModel):
    action: str

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global env_instance
    data = TASKS.get(req.task, TASKS["mixed"])
    env_instance = NotificationEnv(data=data)
    obs = env_instance.reset()
    return {"observation": obs, "done": False}

@app.post("/step")

def step(req: StepRequest):
    global env_instance

    if env_instance is None:
        return {"error": "Call /reset first"}

    if env_instance.done:
        return {"error": "Episode done. Call /reset again."}

    action_obj = Action(mode=req.action)
    obs, reward, done = env_instance.step(action_obj)
    return {
        "observation": obs,
        "action": req.action,
        "reward": round(reward, 4),
        "done": done,
    }
@app.get("/state")
def state():
    global env_instance

    if env_instance is None:
        return {"error": "Call /reset first"}

    return {
        "current_index": env_instance.current_index,
        "total_reward": round(env_instance.total_reward, 4),
        "done": env_instance.done,
    }

@app.get("/")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

