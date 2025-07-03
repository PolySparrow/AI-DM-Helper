import random
import json
import random
import requests
import logging_function
import os
logger = logging_function.setup_logger()

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
def format_history_for_llama(history):
    role_map = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant"
    }
    prompt = ""
    for msg in history:
        role = role_map.get(msg["role"], msg["role"].capitalize())
        content = msg["content"]
        prompt += f"{role}: {content}\n"
    prompt += "Assistant: "
    logger.debug("Formatted prompt for Llama:\n" + prompt)
    return prompt

def get_latest_user_message(history):
    for msg in reversed(history):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""

def roll_labeled_dice_text_from_json(json_input):
    """
    json_input: JSON string representing a list of dice groups.
    Each dice group is a dict with:
      - 'qty': number of dice to roll
      - 'sides': number of sides on the dice
      - 'labels': list of labels for each die (optional)
      - 'sign': +1 or -1 (optional, default +1)
    
    Returns a string showing each roll with label and the total sum.
    """
    dice_groups = json.loads(json_input)
    
    total = 0
    lines = []
    
    for group in dice_groups:
        qty = group['qty']
        sides = group['sides']
        labels = group.get('labels', [None] * qty)
        sign = group.get('sign', 1)
        
        for i in range(qty):
            roll = random.randint(1, sides)
            total += sign * roll
            label = labels[i] if i < len(labels) else f"{qty}d{sides}_die{i+1}"
            sign_str = '-' if sign == -1 else ''
            lines.append(f"{label}: {sign_str}{roll}")
    
    lines.append(f"Total sum: {total}")
    return "\n".join(lines)


def summarize_history_with_llm(history, model=OLLAMA_MODEL):
    # Use your LLM to summarize the history (excluding the last N messages)
    prompt = (
        "Summarize the following conversation in 2-3 sentences for context:\n\n"
        + "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in history)
    )
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def build_hybrid_history(history, max_recent=10, model=OLLAMA_MODEL):
    if len(history) > max_recent + 2:
        summary = summarize_history_with_llm(history[:-max_recent], model=model)
        # Compose a new history: system, summary, last N messages
        hybrid_history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": f"Summary so far: {summary}"},
        ] + history[-max_recent:]
        return hybrid_history
    else:
        return history
    


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "source")

def find_kb_file(kb_name, source_dir=SOURCE_DIR):
    """
    Returns the full path to the file in source_dir whose base name matches kb_name,
    or None if not found.
    """
    for fname in os.listdir(source_dir):
        base, ext = os.path.splitext(fname)
        logger.debug(f"Checking file: {fname}, base: {base}, ext: {ext}")
        if base == kb_name and ext.lower() in [".txt", ".pdf", ".docx", ".csv", ".xlsx"]:
            return os.path.join(source_dir, fname)
    return None
# Example usage:
# json_input = '''
# [
#     {"qty": 2, "sides": 12, "labels": ["hope", "fear"]},
#     {"qty": 1, "sides": 6}
# ]
# '''

# result_text = roll_labeled_dice_text_from_json(json_input)
# print(result_text)