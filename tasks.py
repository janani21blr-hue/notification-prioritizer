import random

# Task 1: Only important notifications
def task_urgent(data):
    return [n for n in data if n["label"] == "important"]


# Task 2: Mixed (original dataset but shuffled)
def task_mixed(data):
    mixed = data.copy()
    random.shuffle(mixed)
    return mixed


# Task 3: Noisy - mostly non-important, but some important hidden inside
def task_noisy(data):
    important = [n for n in data if n["label"] == "important"]
    non_important = [n for n in data if n["label"] != "important"]

    # Keep ~25% of important notifications (minimum 1)
    hidden = important[:max(1, len(important) // 4)]

    mixed = non_important + hidden
    random.shuffle(mixed)   

    return mixed