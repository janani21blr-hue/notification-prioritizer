import json
import random
from data import NOTIFICATIONS
from models import NotificationObservation
from agent import choose_action

# --- CONFIG ---
NUM_SAMPLES = 800  # Good amount for a tiny model like Qwen-0.5B
OUTPUT_FILE = "train_data.jsonl"
STATES = ["studying", "relaxing", "sleeping", "working", "commuting"]

def generate_sample():
    # 1. Pick a random notification
    notif = random.choice(NOTIFICATIONS)
    
    # 2. Randomize the context
    user_state = random.choice(STATES)
    current_focus = round(random.uniform(0.05, 1.0), 2)
    is_annoyed = current_focus < 0.3

    # 3. Create a mock observation object for the agent
    obs = NotificationObservation(
        id=notif["id"],
        app=notif["app"],
        message=notif["message"],
        sender=notif["sender"],
        user_state=user_state,
        current_focus=current_focus,
        is_user_annoyed=is_annoyed
    )

    # 4. Get the "Gold Standard" action from our logic-based agent
    action = choose_action(obs)

    # 5. Format for LLM Fine-tuning (Instruction-Input-Output)
    # We want the model to learn to reason about the focus budget.
    instruction = "You are an AI Attention Guard. Decide if this notification should be: notify, delay, or ignore."
    input_text = (
        f"Context: User is {user_state}. Focus Budget: {int(current_focus*100)}%. "
        f"Annoyed: {is_annoyed}. Notification from {obs.app}: {obs.message}"
    )
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": action
    }

def main():
    print(f"Generating {NUM_SAMPLES} synthetic training samples...")
    dataset = []
    for _ in range(NUM_SAMPLES):
        dataset.append(generate_sample())

    with open(OUTPUT_FILE, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Done! Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()