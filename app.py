# app.py (Root Folder)
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional

from env import NotificationEnv
from data import NOTIFICATIONS
from models import NotificationAction as Action
from tasks import task_urgent, task_mixed, task_noisy
from agent import agent_step

app = FastAPI()

env_instance: Optional[NotificationEnv] = None

TASKS = {
    "urgent": task_urgent(NOTIFICATIONS),
    "mixed":  task_mixed(NOTIFICATIONS),
    "noisy":  task_noisy(NOTIFICATIONS),
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
    # obs is a NotificationObservation object
    return {"observation": obs.dict(), "done": False}

@app.post("/step")
def step(req: StepRequest):
    global env_instance
    if env_instance is None or env_instance.done:
        return {"error": "Reset required", "reward": 0.5}

    action_obj = Action(mode=req.action)
    obs, reward, done = env_instance.step(action_obj)

    return {
        "observation": obs.dict() if obs else None,
        "reward": round(reward, 4),
        "done": done
    }

@app.post("/agent-step")
def agent_step_api():
    global env_instance
    if env_instance is None or env_instance.done:
        return {"error": "Reset required"}

    # Get current observation object
    obs_obj = env_instance._get_observation()
    
    # Agent decision (now focus-aware)
    action_str, _ = agent_step(obs_obj)

    # Step the environment
    action_obj = Action(mode=action_str)
    next_obs, env_reward, done = env_instance.step(action_obj)

    return {
        "previous_observation": obs_obj.dict(),
        "observation": next_obs.dict() if next_obs else None,
        "agent_action": action_str,
        "reward": round(env_reward, 4),
        "done": done
    }

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/")
def root(): return RedirectResponse(url="/ui")

@app.get("/ui", response_class=HTMLResponse)
def ui():
    # I have updated the JS in this HTML to handle the new observation structure
    return """
<!DOCTYPE html>
<html>
<head>
    <title>🔔 Attention Guard Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0b0b0e; color: #e0e0e0; max-width: 800px; margin: 40px auto; padding: 20px; }
        .card { background: #16161d; padding: 20px; border-radius: 12px; margin-top: 15px; border: 1px solid #2d2d35; }
        .focus-bar-bg { background: #333; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        #focus-bar { height: 100%; width: 100%; background: #10b981; transition: width 0.4s ease; }
        .btn-agent { background: #6366f1; color: white; width: 100%; padding: 15px; border-radius: 8px; border: none; cursor: pointer; font-weight: bold; }
        .btn-act { padding: 8px 15px; border-radius: 5px; border: none; cursor: pointer; color: white; margin-right: 5px; font-weight:bold; }
        .notify { background: #10b981; } .delay { background: #f59e0b; } .ignore { background: #ef4444; }
        .label { color: #888; font-size: 0.85em; }
        #log { background: #000; padding: 10px; margin-top: 20px; height: 100px; overflow-y: auto; font-family: monospace; font-size: 0.8em; border-radius: 5px; color: #00ff00; }
    </style>
</head>
<body>
    <h2>🔔 Attention Guard <span style="font-size: 0.5em; color: #6366f1;">v2.0</span></h2>
    
    <div class="card">
        <select id="task" style="background:#222; color:white; padding:5px; border-radius:5px;">
            <option value="urgent">Urgent</option>
            <option value="mixed" selected>Mixed</option>
            <option value="noisy">Noisy</option>
        </select>
        <button onclick="reset()" style="background:#3b82f6; color:white; border:none; padding:6px 12px; border-radius:5px; cursor:pointer;">Reset Environment</button>
    </div>

    <div class="card">
        <div style="display:flex; justify-content:space-between;">
            <span class="label">USER FOCUS BUDGET</span>
            <span id="focus-text" style="font-weight:bold;">100%</span>
        </div>
        <div class="focus-bar-bg"><div id="focus-bar"></div></div>
    </div>

    <div id="obs-card" class="card" style="display:none;">
        <div class="label">Incoming from <span id="obs-app" style="color:white; font-weight:bold;"></span></div>
        <div id="obs-msg" style="font-size: 1.1em; margin: 10px 0;"></div>
        <div class="label">User State: <span id="obs-state" style="color:white;"></span></div>
    </div>

    <div class="card">
        <button class="btn-agent" onclick="agentAct()">🤖 Ask Agent to Decide</button>
        <div style="margin-top:15px; display:flex; gap:10px;">
            <button class="btn-act notify" onclick="act('notify')">Notify</button>
            <button class="btn-act delay" onclick="act('delay')">Delay</button>
            <button class="btn-act ignore" onclick="act('ignore')">Ignore</button>
        </div>
    </div>

    <div id="log">> System Ready. Click Reset to start.</div>

    <script>
        function addLog(m) { const l = document.getElementById('log'); l.innerHTML = `> ${m}<br>` + l.innerHTML; }
        
        function updateUI(data) {
            if (!data.observation) return;
            const obs = data.observation;
            document.getElementById('obs-card').style.display = 'block';
            document.getElementById('obs-app').textContent = obs.app;
            document.getElementById('obs-msg').textContent = obs.message;
            document.getElementById('obs-state').textContent = obs.user_state;
            
            const focus = obs.current_focus * 100;
            document.getElementById('focus-bar').style.width = focus + '%';
            document.getElementById('focus-text').textContent = Math.round(focus) + '%';
            
            const bar = document.getElementById('focus-bar');
            if (focus > 60) bar.style.background = '#10b981';
            else if (focus > 30) bar.style.background = '#f59e0b';
            else bar.style.background = '#ef4444';
            
            if (data.done) addLog("🏁 Episode Finished!");
        }

        async function reset() {
            const task = document.getElementById('task').value;
            const res = await fetch('/reset', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({task}) });
            const data = await res.json();
            updateUI(data);
            addLog(`Reset to task: ${task}`);
        }

        async function agentAct() {
            const res = await fetch('/agent-step', { method: 'POST' });
            const data = await res.json();
            if(data.error) return alert(data.error);
            updateUI(data);
            addLog(`Agent: ${data.agent_action.toUpperCase()} (Reward: ${data.reward})`);
        }

        async function act(choice) {
            const res = await fetch('/step', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({action: choice}) });
            const data = await res.json();
            if(data.error) return alert(data.error);
            updateUI(data);
            addLog(`Manual: ${choice} (Reward: ${data.reward})`);
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)