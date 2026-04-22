# inference.py
import requests
import os

MODEL_ID = "DigitalPixie/attention-guard-v2-brain-f16"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set this in your env

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

def classify_importance(app: str, sender: str, message: str, user_state: str) -> str:
    prompt = f"""Classify this notification importance as one word only (critical/high/medium/low).

App: {app}
Sender: {sender}  
Message: {message}
User State: {user_state}

critical=emergency/OTP/bank, high=interview/offer/exam, medium=friends/social, low=promo/marketing

Answer (one word):"""

    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 5, "return_full_text": False}},
            timeout=10
        )
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("generated_text", "").strip().lower()
            for word in text.split():
                if word in ["critical", "high", "medium", "low"]:
                    return word
    except Exception as e:
        print(f"HF API call failed: {e}")
    
    return _keyword_fallback(app, sender, message)

def _keyword_fallback(app: str, sender: str, message: str) -> str:
    msg = message.lower()
    sender_l = sender.lower()
    if any(k in msg for k in ["emergency", "urgent", "otp", "bank", "security"]):
        return "critical"
    if any(k in msg for k in ["interview", "offer", "exam", "deadline", "shortlisted"]):
        return "high"
    if any(k in msg for k in ["sale", "discount", "promo", "liked", "reel", "story"]):
        return "low"
    if sender_l in ["mom", "dad", "placement cell"]:
        return "high"
    return "medium"