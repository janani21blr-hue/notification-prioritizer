import random

q_table = {}
episode_log = []

ALPHA   = 0.2
GAMMA   = 0.9
EPSILON = 0.05

ACTIONS = ["notify", "delay", "ignore"]

# ============================================================
# STEP 1: Classify message importance
# ============================================================
def get_importance(obs):
    message = obs.get("message", "").lower()
    sender  = obs.get("sender",  "").lower()
    app     = obs.get("app", obs.get("App", "")).lower()

    # --- CRITICAL ---
    critical_keywords = [
        "otp", "bank", "transaction", "emergency", "accident", "fire",
        "hospital", "police", "call now", "sos", "help me",
        "security alert", "login attempt", "fraud"
    ]
    if any(k in message for k in critical_keywords):
        return "critical"

    family_senders = ["mom", "dad", "amma", "appa", "mother", "father", "parents"]
    call_keywords  = ["calling you", "is calling", "call me", "called you"]
    if any(s in sender for s in family_senders) and any(c in message for c in call_keywords):
        return "critical"
    
    # --- HIGH ---
    high_keywords = [
        "deadline", "urgent", "exam", "result", "interview",
        "assignment due", "due in", "shortlisted", "selected",
        "offer letter", "payment due", "verification", "internship",
        "job offer", "hall ticket", "admit card", "submission deadline",
        "in 2 hours", "in 1 hour", "expires soon", "valid for","asap","immediately"
    ]
    if any(w in message for w in high_keywords):
        if any(t in message for t in ["tomorrow", "next week", "later", "no rush"]):
            return "medium"
        return "high"

    high_senders = ["prof", "college", "hr@", "recruiter", "bank@", "admin", "noreply@", "support@"]
    if any(s in sender for s in high_senders):
        return "high"

    # --- LOW ---
    low_keywords = ["sale", "discount", "offer", "promo", "free", "deal", "30%", "off", "cashback", "coupon", "flat","%off"]
    if any(w in message for w in low_keywords):
        return "low"

    low_apps = ["youtube", "netflix", "instagram", "spotify", "swiggy", "zomato", "amazon", "flipkart"]
    delivery_keywords = ["out for delivery", "arriving", "delivered", "order placed", "order confirmed"]
    if any(a in app for a in low_apps):
        if not any(d in message for d in delivery_keywords):
            return "low"

    # --- MEDIUM ---
    medium_keywords = ["are you coming", "what are you", "when are you", "let's meet", "plan", "badminton", "cricket", "match tomorrow"]
    if any(w in message for w in medium_keywords):
        return "medium"

    family_or_friends = ["mom", "dad", "friend", "bro", "sis", "dude"]
    if any(s in sender for s in family_or_friends):
        return "medium"

    return "medium"

# ============================================================
# STEP 2: User state
# ============================================================
def get_user_state_bucket(user_state):
    us = user_state.lower().strip()
    if any(s in us for s in ["sleeping", "asleep"]): return "sleeping"
    if any(s in us for s in ["meeting", "busy", "dnd"]): return "unavailable"
    if any(s in us for s in ["studying", "working", "focused", "coding"]): return "focused"
    if any(s in us for s in ["relaxing", "free", "idle"]): return "free"
    return "neutral"

# ============================================================
# STEP 3: Decision matrix
# ============================================================
DECISION_MATRIX = {
    ("critical", "sleeping"): "notify", ("critical", "unavailable"): "notify", ("critical", "focused"): "notify", ("critical", "free"): "notify", ("critical", "neutral"): "notify",
    ("high", "sleeping"): "delay", ("high", "unavailable"): "notify", ("high", "focused"): "notify", ("high", "free"): "notify", ("high", "neutral"): "notify",
    ("medium", "sleeping"): "delay", ("medium", "unavailable"): "delay", ("medium", "focused"): "delay", ("medium", "free"): "notify", ("medium", "neutral"): "notify",
    ("low", "sleeping"): "ignore", ("low", "unavailable"): "ignore", ("low", "focused"): "ignore", ("low", "free"): "ignore", ("low", "neutral"): "ignore",
}

# ============================================================
# REWARD TABLE — Boundary safe (0.01 to 0.99)
# ============================================================
REWARD_TABLE = {
    # CRITICAL
    ("critical", "sleeping", "notify"): 0.99, ("critical", "sleeping", "delay"): 0.3, ("critical", "sleeping", "ignore"): 0.01,
    ("critical", "unavailable", "notify"): 0.99, ("critical", "unavailable", "delay"): 0.3, ("critical", "unavailable", "ignore"): 0.01,
    ("critical", "focused", "notify"): 0.99, ("critical", "focused", "delay"): 0.3, ("critical", "focused", "ignore"): 0.01,
    ("critical", "free", "notify"): 0.99, ("critical", "free", "delay"): 0.3, ("critical", "free", "ignore"): 0.01,
    ("critical", "neutral", "notify"): 0.99, ("critical", "neutral", "delay"): 0.3, ("critical", "neutral", "ignore"): 0.01,

    # HIGH
    ("high", "sleeping", "notify"): 0.7, ("high", "sleeping", "delay"): 0.99, ("high", "sleeping", "ignore"): 0.1,
    ("high", "unavailable", "notify"): 0.99, ("high", "unavailable", "delay"): 0.6, ("high", "unavailable", "ignore"): 0.1,
    ("high", "focused", "notify"): 0.99, ("high", "focused", "delay"): 0.5, ("high", "focused", "ignore"): 0.1,
    ("high", "free", "notify"): 0.99, ("high", "free", "delay"): 0.4, ("high", "free", "ignore"): 0.1,
    ("high", "neutral", "notify"): 0.99, ("high", "neutral", "delay"): 0.5, ("high", "neutral", "ignore"): 0.1,

    # MEDIUM
    ("medium", "sleeping", "notify"): 0.2, ("medium", "sleeping", "delay"): 0.99, ("medium", "sleeping", "ignore"): 0.5,
    ("medium", "unavailable", "notify"): 0.4, ("medium", "unavailable", "delay"): 0.99, ("medium", "unavailable", "ignore"): 0.5,
    ("medium", "focused", "notify"): 0.3, ("medium", "focused", "delay"): 0.99, ("medium", "focused", "ignore"): 0.4,
    ("medium", "free", "notify"): 0.99, ("medium", "free", "delay"): 0.6, ("medium", "free", "ignore"): 0.2,
    ("medium", "neutral", "notify"): 0.99, ("medium", "neutral", "delay"): 0.6, ("medium", "neutral", "ignore"): 0.2,

    # LOW
    ("low", "sleeping", "notify"): 0.01, ("low", "sleeping", "delay"): 0.5, ("low", "sleeping", "ignore"): 0.99,
    ("low", "unavailable", "notify"): 0.01, ("low", "unavailable", "delay"): 0.4, ("low", "unavailable", "ignore"): 0.99,
    ("low", "focused", "notify"): 0.01, ("low", "focused", "delay"): 0.3, ("low", "focused", "ignore"): 0.99,
    ("low", "free", "notify"): 0.1, ("low", "free", "delay"): 0.5, ("low", "free", "ignore"): 0.99,
    ("low", "neutral", "notify"): 0.1, ("low", "neutral", "delay"): 0.6, ("low", "neutral", "ignore"): 0.99,
}

def get_state_key(obs):
    importance = get_importance(obs)
    user_state = obs.get("user_state", obs.get("User State", "unknown"))
    state_bucket = get_user_state_bucket(user_state)
    return f"{importance}|{state_bucket}"

def choose_action(state_key, obs):
    importance = get_importance(obs)
    user_state = obs.get("user_state", obs.get("User State", ""))
    state_bucket = get_user_state_bucket(user_state)
    rule_action = DECISION_MATRIX.get((importance, state_bucket), "delay")

    if state_key in q_table:
        values = q_table[state_key]
        best_action = max(values, key=values.get)
        if values[best_action] > 0.8: return best_action

    if random.random() < EPSILON: return random.choice(ACTIONS)
    return rule_action

def get_agent_reward(action, importance, user_state):
    state_bucket = get_user_state_bucket(user_state)
    # Clamp the lookup to be doubly safe
    raw_reward = REWARD_TABLE.get((importance, state_bucket, action), 0.5)
    return float(max(0.01, min(0.99, raw_reward)))

def update_q_table(state_key, action, reward):
    if state_key not in q_table:
        q_table[state_key] = {"notify": 0.0, "delay": 0.0, "ignore": 0.0}
    old_val = q_table[state_key][action]
    best_next = max(q_table[state_key].values())
    q_table[state_key][action] = old_val + ALPHA * (reward + GAMMA * best_next - old_val)

def agent_step(obs):
    state_key  = get_state_key(obs)
    action     = choose_action(state_key, obs)
    user_state = obs.get("user_state", obs.get("User State", ""))
    importance = get_importance(obs)
    reward     = get_agent_reward(action, importance, user_state)
    update_q_table(state_key, action, reward)
    return action, reward