def smart_agent_action(obs):
    if obs is None:
            return "ignore"
    message = obs.get("message", "").lower()
    app = obs.get("app", "").lower()
    user_state = obs.get("user_state", "relaxing")

    score = 0

    # --- HIGH PRIORITY keywords ---
    if any(word in message for word in [
        "exam", "deadline", "otp", "interview", "shortlisted",
        "assignment", "meeting", "due", "reminder", "gate"
    ]):
        score += 6

    # --- MEDIUM PRIORITY ---
    if any(word in message for word in [
        "delivery", "calling", "reschedule"
    ]):
        score += 3

    # --- LOW PRIORITY ---
    if any(word in message for word in [
        "off", "discount", "sale", "liked", "video", "restaurant", "netflix"
    ]):
        score -= 4

    # --- APP BASED SIGNAL ---
    if app == "gmail":
        score += 3
    elif app == "whatsapp":
        score += 1
    elif app in ["zomato", "swiggy", "youtube", "instagram"]:
        score -= 3

    # --- DECISION LOGIC ---
    if user_state == "studying":
        if score >= 6:
            return "notify"
        elif score >= 2:
            return "delay"
        else:
            return "ignore"

    else:  # relaxing
        if score >= 5:
            return "notify"
        elif score >= 1:
            return "delay"
        else:
            return "ignore"