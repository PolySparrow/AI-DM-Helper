import random
import json
import random
import requests
from logging_function import setup_logger
import os
from environment_vars import OLLAMA_URL, OLLAMA_MODEL, SOURCE_DIR
import logging
from rake_nltk import Rake
from keybert import KeyBERT
import spacy

setup_logger(app_name="AI_DM_RAG")  # or whatever app name you want
logger = logging.getLogger(__name__)



rake = Rake()
kw_model = KeyBERT()
nlp = spacy.load("en_core_web_sm")

def extract_tags(text, top_n=5):
    tags = set()
    # RAKE
    rake.extract_keywords_from_text(text)
    tags.update(rake.get_ranked_phrases()[:top_n])
    # KeyBERT
    try:
        keybert_keywords = kw_model.extract_keywords(text, top_n=top_n)
        tags.update([kw for kw, _ in keybert_keywords])
    except Exception as e:
        print(f"KeyBERT failed: {e}")
    # spaCy noun chunks and entities
    doc = nlp(text)
    tags.update([chunk.text for chunk in doc.noun_chunks])
    tags.update([ent.text for ent in doc.ents])
    # Clean up
    tags = {t.strip().lower() for t in tags if t.strip()}
    return list(tags)

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