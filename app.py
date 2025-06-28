import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import config
import chatbot_functions
import logging_function

logger = logging_function.setup_logger()

class SmartOpenAIChat:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        # If you don't use function calling, set self.functions = None
        self.functions = getattr(chatbot_functions, "functions", None)
        logger.debug(f"Initialized SmartOpenAIChat with model {self.model}")

    def handle(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=self.functions,
                function_call="auto" if self.functions else None,
                max_tokens=256,
                temperature=0
            )
            message = response.choices[0].message
            logger.debug(f"Received response from LLM: {message}")

            # If a function call is returned, handle it (optional)
            if hasattr(message, "function_call") and message.function_call:
                fn_name = message.function_call.name
                args = json.loads(message.function_call.arguments)
                try:
                    endpoint = f"http://127.0.0.1:5001/api/v1.0/{fn_name}"
                    logger.debug(f"Calling function {fn_name} with arguments {args} at endpoint {endpoint}")
                    returned_data = self.send_request(endpoint, args)
                    results = returned_data.get("results")
                    return f"I used this function {fn_name} with arguments {args} and got the response: \n\n {results}", fn_name
                except Exception as e:
                    return f"Error calling function {fn_name}: {str(e)}", fn_name
            else:
                # Just return the LLM's message
                return message.content, "chat"
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return f"Error processing request: {str(e)}", "chat"

    def send_request(self, endpoint, args):
        # Dummy implementation; replace with your actual function call logic
        import requests
        response = requests.post(endpoint, json=args)
        print("Status code:", response.status_code)
        print("Raw response:", response.text)
        return response.json()

# Flask app setup
app = Flask(__name__)
CORS(app)

chatbot = SmartOpenAIChat(
    api_key=config.openai_apikey,
    model=os.getenv("OPENAI_MODEL", "gpt-4o")
)

@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    history = data.get('history', [])
    logger.debug(f"Received history: {history}")
    if not history or not isinstance(history, list):
        return jsonify({'success': False, 'error': 'No prompt provided'}), 400

    # Convert history to OpenAI's expected format
    messages = []
    for msg in history:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue  # skip malformed messages
        # Ensure content is a string
        if not isinstance(msg["content"], str):
            continue
        messages.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}]
        })

    if not messages:
        return jsonify({'success': False, 'error': 'No valid messages in history'}), 400

    try:
        result, fn_name = chatbot.handle(messages)
        return jsonify({'success': True, 'response': result, 'function': fn_name})
    except Exception as e:
        logger.error(f"Exception in chat_route: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    logger.debug("Starting the SmartOpenAIChat Flask app...")
    app.run(debug=True, host='127.0.0.1', port=5000)