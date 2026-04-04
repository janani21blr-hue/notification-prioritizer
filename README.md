## 🔔 Notification Prioritizer

A decision-driven notification system that optimizes when to interrupt, delay, or ignore — based on context, not just importance.
A context-aware system that decides whether to notify, delay, or ignore notifications based on user state and content.

Unlike traditional binary filtering, this models notification handling as a **sequential decision problem**, enabling smarter trade-offs.

- Handles real-world ambiguity  
- Adapts to user context (studying vs relaxing)  
- Robust to noisy notifications

## 📊 Results

| Task   | Random | Smart Agent| Improvement|
|--------|--------|------------|------------|
| Urgent | 3.42   | 7.3        | +113%|
| Mixed  | 9.17   | 15.7       | +71% |
| Noisy  | 6.92   | 10.4       | +50% |

## 💡 Key Idea

Instead of predicting importance, the agent learns **what action maximizes long-term reward**.

This enables:
- smarter handling of optional notifications
- balancing interruption vs usefulness
- better behavior in noisy environments

##  🎯 Motivation

Notifications are not binary (important vs not).  
Their relevance depends on **context** and **timing**.

Example:
- Delivery update → irrelevant while studying, useful while relaxing  
- Social message → delay instead of ignore

##  ⚙️ Environment Design
### 👁️ Observation Space

The agent receives one notification at a time as a dictionary:

| Field | Type | Description |
|---|---|---|
| id | int | Unique notification ID |
| app | str | Source app (Gmail, WhatsApp, Swiggy, etc.) |
| message | str | Notification text |
| sender | str | Sender identity |
| user_state | str | Current user context: `studying` or `relaxing` |

The label (important / optional / ignore) is **hidden from the agent** — it is only used internally to compute the reward.

---
### 🎮 Action Space

| Action | Meaning |
|---|---|
| `notify` | Interrupt the user immediately |
| `delay` | Queue for later |
| `ignore` | Suppress entirely |

These three actions reflect real notification management behavior and create non-trivial tradeoffs — for optional notifications, `delay` outperforms both `notify` and `ignore`.

---
##  🧮 Reward Function

Rewards are continuous in `[0.0, 1.0]` based on the `(action, label)` pair:

| Action | important | optional | ignore |
|---     |---        |---       |---     |
| notify | 1.0       | 0.7      | 0.0    |
| delay  | 0.3       | 0.8      | 0.5    |
| ignore | 0.0       | 0.5      | 1.0    |

**Design intent:** Optional notifications have no single perfect action — `delay` is best, but `notify` and `ignore` both give partial reward. This forces the agent to reason beyond binary classification.

---

### 🔄 user_state Modifiers

| Condition | Modifier |
|---|---|
| `studying` + `notify` + label=`ignore` | −0.2 (penalty for junk interruption) |
| `studying` + `ignore` + label=`important` | −0.1 (penalty for missing critical alert) |
| `relaxing` + `delay` + label=`optional` | +0.1 (bonus for appropriate deferral) |

These modifiers keep all rewards within `[0.0, 1.0]` and make `user_state` a meaningful signal rather than a decorative field.
---
##  🧪 Tasks
### 1. `task_urgent` — Precision Under Pressure
Dataset contains only `important` notifications. The agent should learn to consistently notify high-priority signals.

### 2. `task_mixed` — Balanced Decision-Making
The full real-world dataset with all label types.

### 3. `task_noisy` — Robustness Under Noise
Predominantly low-priority notifications with ~25% important ones mixed in.
---

## 📉 Baseline

- Random Agent: selects actions randomly  
- Rule-based Agent: keyword + app-based heuristic

## 🤖 Example Behavior

- Notify urgent academic messages while studying  
- Ignore promotional spam during focus time  
- Delay casual/social messages  
- Always notify critical alerts (e.g., OTPs)

## 📁 Project Structure (Modular Design)

Notification-Prioritizer/
├── env.py            # Environment logic (step, reset, observation)
├── models.py         # Action + observation models
├── tasks.py          # Task definitions (urgent, mixed, noisy)
├── rewards.py        # Reward function
├── baseline.py       # Evaluation script
├── data.py           # Notification dataset
├── app.py            # FastAPI interface
├── inference.py      # Agent inference logic
├── requirements.txt  # Dependencies
├── Dockerfile        # Containerization

---

## 🚀 Setup & Running
### Local
pip install -r requirements.txt
python baseline.py

###  Docker

docker build -t notification-prioritizer .
docker run notification-prioritizer

---

## 🧠 Design Highlights

- 3 actions (notify, delay, ignore) enable better real-world decisions than binary filtering  
- Partial rewards capture ambiguity in optional notifications  
- User state modifies reward, making context meaningful  
- Rule-based agent is strong but fails in ambiguous cases

## 🔮 Future Work

- Personalization per user  
- Learning-based agents (RL)  
- Richer context (time, history)



