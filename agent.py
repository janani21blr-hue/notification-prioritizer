def get_importance(obs) -> str:
    """
    Uses fine-tuned Qwen LLM to classify importance.
    Falls back to label during training (when _label is injected).
    """
    # During training, use injected label directly (fast, no LLM needed)
    if hasattr(obs, "_label") and obs._label:
        label = obs._label
        if label == "important":
            return "critical"
        elif label == "optional":
            return "medium"
        else:
            return "low"

    # During inference, use the fine-tuned LLM
    try:
        from inference import classify_importance
        return classify_importance(obs.app, obs.sender, obs.message, obs.user_state)
    except Exception as e:
        print(f"LLM inference failed, using fallback: {e}")
        # Keyword fallback
        msg = obs.message.lower()
        sender_l = obs.sender.lower()
        if any(k in msg for k in ["emergency", "urgent", "otp", "bank", "security"]):
            return "critical"
        if any(k in msg for k in ["interview", "offer", "exam", "deadline"]):
            return "high"
        if any(k in msg for k in ["sale", "discount", "promo", "liked"]):
            return "low"
        if sender_l in ["mom", "dad", "placement cell"]:
            return "high"
        return "medium"