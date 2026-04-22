import random

TEMPLATES = [
    # CRITICAL (Weight: 30%)
    ({"app": "WhatsApp", "sender": "Mom", "label": "important", "msgs": ["Pick up the phone", "Emergency call me"]}, 15),
    ({"app": "Gmail", "sender": "Placement Cell", "label": "important", "msgs": ["Interview link attached", "Offer letter"]}, 15),
    
    # OPTIONAL (Weight: 30%)
    ({"app": "WhatsApp", "sender": "Sneha", "label": "optional", "msgs": ["Badminton tomorrow?", "Lunch plans?"]}, 15),
    ({"app": "Instagram", "sender": "Alex", "label": "optional", "msgs": ["Sent you a reel", "Liked your story"]}, 15),
    
    # JUNK (Weight: 40%)
    ({"app": "Swiggy", "sender": "Promo", "label": "junk", "msgs": ["50% off pizza!", "Free delivery!"]}, 20),
    ({"app": "Gmail", "sender": "Marketing", "label": "junk", "msgs": ["Clearance sale ends today"]}, 20)
]

def generate_synthetic_data(count=50):
    dataset = []
    templates, weights = zip(*TEMPLATES)
    for i in range(count):
        # Programmatically select based on real-world weights
        template = random.choices(templates, weights=weights, k=1)[0]
        dataset.append({
            "id": i,
            "app": template["app"],
            "sender": template["sender"],
            "message": random.choice(template["msgs"]),
            "label": template["label"],
            "user_state": random.choice(["studying", "relaxing"])
        })
    return dataset

NOTIFICATIONS = generate_synthetic_data(50)