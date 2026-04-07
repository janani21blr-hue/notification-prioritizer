---
title: Notification Prioritizer
emoji: 🔔
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
---
🔔 Notification Prioritizer — OpenEnv Hackathon Submission

A context-aware reinforcement learning environment where an agent learns to intelligently decide: notify, delay, or ignore — based on who's messaging, what they're saying, and what the user is doing right now.


🧠 The Problem
Modern smartphones deliver dozens of notifications per hour. Most are irrelevant to what the user is doing at that moment. A message from a friend while you're in a meeting, a promotional alert while you're studying, a spam notification while you're sleeping — these don't just annoy; they fragment attention and reduce productivity.
Notification overload is a real cognitive cost. This project builds an agent that solves it.

💡 The Solution
A sequential decision-making environment where an agent observes a notification's context and chooses the best action:
ActionMeaningnotifyShow it immediatelydelayHold it for laterignoreSuppress it entirely
The agent uses an adaptive decision policy approximating optimal behavior via reward signals — evaluating app type, message urgency, sender relationship, and current user state before acting.

🌍 Environment Design
Observation Space
Each step provides the agent with:
json{
  "id": 12,
  "app": "WhatsApp",
  "message": "Your OTP is 847291",
  "sender": "Bank",
  "user_state": "studying"
}
Action Space
0 → notify
1 → delay  
2 → ignore
Reward Logic
Rewards are context-sensitive, not rule-based. The same action can yield different rewards depending on urgency, sender, and user state. Optional/social notifications intentionally have no single "perfect" action — the agent must learn tradeoffs.

📊 Baseline Evaluation Results
The Smart Agent (context-aware decision policy) was evaluated against a Random Agent across three task difficulties:
TaskRandom AgentSmart AgentImprovementURGENT3.427.30+113.14%MIXED9.1715.70+71.21%NOISY6.9210.40+50.18%
The agent performs strongest on urgent tasks (where correct prioritization has the highest payoff) and remains significantly better than random even in noisy environments with ambiguous notifications.

🏗️ Project Structure
notification-prioritizer/
├── app.py            # FastAPI server — exposes /reset, /step, /state
├── env.py            # Core environment logic (Gym-style)
├── data.py           # Notification dataset (30 diverse samples)
├── rewards.py        # Reward table with context-sensitive scoring
├── agent.py          # Smart agent — decision policy based on contextual scoring
├── evaluate.py       # Evaluation runner (baseline comparisons)
├── baseline.py       # Random vs Smart agent benchmark
├── inference.py      # OpenAI-compatible inference with structured logging
├── models.py         # Pydantic request/response schemas
├── tasks.py          # Task configurations (urgent / mixed / noisy)
├── openenv.yaml      # Environment specification
├── Dockerfile        # Container definition
└── requirements.txt  # Dependencies

🚀 API Endpoints
Deployed at: https://digitalpixie-notification-prioritizer.hf.space
MethodEndpointDescriptionGET/Health check — returns {"status": "ok"}POST/resetReset environment, get first observationPOST/stepSubmit action, receive reward + next observationGET/stateView current environment state
Example: Reset
bashcurl -X POST https://digitalpixie-notification-prioritizer.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "mixed"}'
Response:
json{
  "observation": {
    "id": 16,
    "app": "WhatsApp",
    "message": "Did you watch that new series on Netflix?",
    "sender": "Rahul",
    "user_state": "studying"
  },
  "done": false
}
Example: Step
bashcurl -X POST https://digitalpixie-notification-prioritizer.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": "ignore"}'
Response:
json{
  "observation": { ... },
  "action": "ignore",
  "reward": 2,
  "done": false
}

🧪 Task Variants
TaskDescriptionurgentMostly time-sensitive notifications — high stakes, clear correct actionsmixedBlend of urgent, optional, and social — agent must balance tradeoffsnoisy~25% important, ~75% low-priority — tests filtering under ambiguity

⚙️ Running Locally
bash# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --reload

# Run baseline evaluation
python baseline.py

# Run inference with structured logs
python inference.py

🐳 Docker
bashdocker build -t notification-prioritizer .
docker run -p 7860:7860 notification-prioritizer

🔧 Environment Variables
VariableDescriptionNEBIUS_API_KEYAPI key for Qwen2.5-72B-Instruct inferenceBASE_URLInference endpoint base URLMODEL_NAMEModel identifier (default: Qwen/Qwen2.5-72B-Instruct)

📝 Inference Logging Format
The inference.py module emits structured logs for every decision:
[START] episode=1 task=mixed
[STEP] obs={"app": "Gmail", ...} action=delay reward=1
[END] total_reward=12 steps=10

🛠️ Built With

FastAPI — REST API server
Uvicorn — ASGI server
Pydantic — Request/response validation
OpenAI SDK — Compatible inference client
Qwen2.5-72B-Instruct — LLM backbone via Nebius AI
HuggingFace Spaces — Deployment platform
Docker — Containerization


🔥 Why This Is Interesting
Most notification systems use static rules: "urgent keywords = notify." This project frames it differently — as a sequential decision problem under uncertainty.

The agent doesn't know the user's actual preference upfront
Rewards are sparse and sometimes ambiguous (optional notifications have no single correct action)
The environment has three distinct difficulty modes, each testing a different failure mode of naive agents
A random agent isn't just "bad" — it's provably outperformed by 50–113% depending on task complexity

This mirrors real-world challenges in attention management, assistive AI, and context-aware computing — where the cost of a wrong decision isn't just missed information, it's broken focus.

🚧 Limitations & Future Work
LimitationPotential FixRule-based reward tableLearn rewards from user feedback (RLHF-style)Fixed notification datasetStream real notifications via OS APIsSingle-user contextMulti-user profiles with personalized policiesNo temporal memoryAdd recurrent state to track notification historyBinary user statesRicher context: calendar, location, device activity
The current implementation is a strong baseline — the architecture is intentionally designed to be extensible toward learned policies (Q-learning, PPO) without changing the environment interface.

👤 Author
Janani — First-year CSE student, NMIT Bangalore
Built solo as part of the OpenEnv Hackathon 2025.

"Every notification is a choice. This agent learns to make that choice wisely."


