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

# 🔔 Notification Prioritizer — OpenEnv Hackathon Submission

> *Every notification is a choice. This agent learns to make that choice wisely.*

A context-aware reinforcement learning environment where an agent learns to intelligently decide: **notify, delay, or ignore** — based on who's messaging, what they're saying, and what the user is doing right now.

Built solo by a first-year CSE student as part of the **Meta PyTorch × Scaler OpenEnv Hackathon 2026**.

---

## 🧠 The Problem

Modern smartphones deliver dozens of notifications per hour. Most are irrelevant to what the user is doing at that moment.

- A message from a friend while you're in a meeting
- A promotional alert while you're studying
- A spam notification while you're sleeping

These don't just annoy — they **fragment attention and reduce productivity**. Notification overload is a real cognitive cost. This project builds an agent that solves it.

---

## 💡 The Solution

A sequential decision-making environment where an agent observes a notification's context and chooses the best action:

| Action | Meaning |
|--------|---------|
| `notify` | Interrupt the user immediately |
| `delay` | Queue it for later |
| `ignore` | Suppress it entirely |

Unlike traditional binary filtering ("important vs not"), this frames notification handling as a **sequential decision problem under uncertainty** — where the cost of a wrong decision isn't just missed information, it's broken focus.

---

## 📊 Baseline Evaluation Results

The Smart Agent (context-aware decision policy) was evaluated against a Random Agent across three task difficulties:

| Task   | Random Agent | Smart Agent | Improvement |
|--------|-------------|-------------|-------------|
| URGENT | 3.42        | 7.30        | **+113.14%** |
| MIXED  | 9.17        | 15.70       | **+71.21%**  |
| NOISY  | 6.92        | 10.40       | **+50.18%**  |

The agent performs strongest on urgent tasks (where correct prioritization has the highest payoff) and remains significantly better than random even in noisy environments with ambiguous notifications.

---

## 🌍 Environment Design

### Observation Space

Each step provides the agent with one notification as a structured dictionary:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Unique notification ID |
| `app` | `str` | Source app (Gmail, WhatsApp, Swiggy, etc.) |
| `message` | `str` | Notification text |
| `sender` | `str` | Sender identity |
| `user_state` | `str` | Current user context (e.g. `studying`, `relaxing`, `sleeping`) |

The label (`important` / `optional` / `junk`) is **hidden from the agent** — used only internally to compute rewards.

---

### Action Space

| Action | Code | Real-world Meaning |
|--------|------|--------------------|
| Notify | `0` | Interrupt the user immediately |
| Delay | `1` | Hold it for later |
| Ignore | `2` | Suppress entirely |

---

### Reward Function

Rewards are continuous in `[0.0, 1.0]` based on the `(action, label)` pair:

| Action | `important` | `optional` | `junk` |
|--------|------------|------------|--------|
| `notify` | 1.0 | 0.7 | 0.0 |
| `delay` | 0.3 | 0.8 | 0.5 |
| `ignore` | 0.0 | 0.5 | 1.0 |

**Design intent:** Optional notifications have no single "perfect" action — `delay` is best, but `notify` and `ignore` both give partial reward. This forces the agent to reason beyond binary classification.

---

### `user_state` Modifiers

| Condition | Modifier |
|-----------|----------|
| `studying` + `notify` + label=`junk` | −0.2 (penalty for junk interruption) |
| `studying` + `ignore` + label=`important` | −0.1 (penalty for missing critical alert) |
| `relaxing` + `delay` + label=`optional` | +0.1 (bonus for appropriate deferral) |

These modifiers keep all rewards within `[0.0, 1.0]` and make `user_state` a meaningful signal rather than a decorative field.

---

## 🧪 Task Variants

| Task | Description | Difficulty |
|------|-------------|------------|
| `urgent` | Mostly time-sensitive notifications — high stakes, clear correct actions | Easy |
| `mixed` | Blend of urgent, optional, and social — agent must balance tradeoffs | Medium |
| `noisy` | ~25% important, ~75% low-priority — tests filtering under ambiguity | Hard |

---

## 🏗️ Project Structure

```
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
```

---

## 🚀 API Endpoints

Deployed at: `https://digitalpixie-notification-prioritizer.hf.space`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — returns `{"status": "ok"}` |
| `POST` | `/reset` | Reset environment, get first observation |
| `POST` | `/step` | Submit action, receive reward + next observation |
| `GET` | `/state` | View current environment state |

**Example: Reset**
```bash
curl -X POST https://digitalpixie-notification-prioritizer.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "mixed"}'
```

Response:
```json
{
  "observation": {
    "id": 16,
    "app": "WhatsApp",
    "message": "Did you watch that new series on Netflix?",
    "sender": "Rahul",
    "user_state": "studying"
  },
  "done": false
}
```

**Example: Step**
```bash
curl -X POST https://digitalpixie-notification-prioritizer.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": "ignore"}'
```

Response:
```json
{
  "observation": { "..." : "..." },
  "action": "ignore",
  "reward": 2,
  "done": false
}
```

---

## ⚙️ Setup & Running

### Local

```bash
pip install -r requirements.txt
uvicorn app:app --reload
python baseline.py
```

### Docker

```bash
docker build -t notification-prioritizer .
docker run -p 7860:7860 notification-prioritizer
```

---

## 🔧 Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM proxy endpoint (injected by evaluator) |
| `API_KEY` | API key for LLM inference (injected by evaluator) |
| `MODEL_NAME` | Model identifier (default: `gpt-4o-mini`) |
| `ENV_BASE_URL` | Notification environment base URL |

---

## 📝 Inference Logging Format

The `inference.py` module emits structured logs for every decision:

```
[START] task=mixed env=notification-prioritizer model=gpt-4o-mini
[STEP] step=1 action=delay reward=0.80 done=false error=null
[END] success=true steps=10 rewards=0.80,1.00,0.50,...
```

---

## 🔥 Why This Is Interesting

Most notification systems use static rules: *"urgent keywords = notify."* This project frames it differently — as a **sequential decision problem under uncertainty**.

- The agent doesn't know the user's actual preference upfront
- Rewards are sparse and sometimes ambiguous (optional notifications have no single correct action)
- The environment has three distinct difficulty modes, each testing a different failure mode of naive agents
- A random agent isn't just "bad" — it's provably outperformed by **50–113%** depending on task complexity

This mirrors real-world challenges in **attention management, assistive AI, and context-aware computing**.

---

## 🛠️ Built With

- **FastAPI** — REST API server
- **Uvicorn** — ASGI server
- **Pydantic** — Request/response validation
- **OpenAI SDK** — Compatible inference client
- **HuggingFace Spaces** — Deployment platform
- **Docker** — Containerization

---

## 🚧 Limitations & Future Work

| Limitation | Potential Fix |
|------------|---------------|
| Rule-based reward table | Learn rewards from user feedback (RLHF-style) |
| Fixed notification dataset | Stream real notifications via OS APIs |
| Single-user context | Multi-user profiles with personalized policies |
| No temporal memory | Add recurrent state to track notification history |
| Binary user states | Richer context: calendar, location, device activity |

---

## 👤 Author

**Janani** — First-year CSE student, NMIT Bangalore
Built solo as part of the OpenEnv Hackathon 2026