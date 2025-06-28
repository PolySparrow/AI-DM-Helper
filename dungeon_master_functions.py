import openai
import logging
import config
import random
import json
import random

def CreateOpenAIClient(openai_apikey=config.openai_apikey):
    """Create and return an OpenAI client."""
    return openai.OpenAI(openai_apikey)

def Chat(prompt):
    AI=CreateOpenAIClient()
    """Send a prompt to the OpenAI API and return the response."""
    try:
        response = AI.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        message=response.choices[0].message.content.strip()
        AI.close()
        return message
    except Exception as e:
        logging.error(f"Error in Chat function: {e}")
        return str(e)



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

# Example usage:
json_input = '''
[
    {"qty": 2, "sides": 12, "labels": ["hope", "fear"]},
    {"qty": 1, "sides": 6}
]
'''

result_text = roll_labeled_dice_text_from_json(json_input)
print(result_text)