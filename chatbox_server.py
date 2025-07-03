from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging_function
logger = logging_function.setup_logger()

DUNGEON_MASTER_API_URL = "http://127.0.0.1:5001/api/v1.0/hybrid_search"

class ChatboxServer:
    def __init__(self, dungeon_master_api_url):
        self.dungeon_master_api_url = dungeon_master_api_url

    def handle_chat(self, history, kb_names):
        # Find the latest user message
        user_query = ""
        for msg in reversed(history):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
        if not user_query:
            return {'success': False, 'error': 'No user query found'}, 400
        if not kb_names:
            return {'success': False, 'error': 'No knowledge bases selected'}, 400

        try:
            dm_response = requests.post(
                self.dungeon_master_api_url,
                json={"history": history, "knowledge_bases": kb_names},
                timeout=360
            )
            logger.debug(f"DM API request payload: {history}, {kb_names}")
            logger.debug(f"DM API request URL: {self.dungeon_master_api_url}")
            logger.info(f"DM API response status: {dm_response.status_code}")
            logger.debug(f"DM API response content: {dm_response.text}")
            dm_data = dm_response.json()
            if dm_data.get("success"):
                return {
                    "success": True,
                    "response": dm_data["response"],
                    "used_kbs": dm_data.get("used_kbs", []),
                    "function": "chat"
                }, 200
            else:
                return {"success": False, "error": dm_data.get("error", "Unknown error")}, 500
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500

knowledge_bases = {
    "core_rules": {},
    "adversaries": {},
    "environments": {},
    "domain_card_reference": {},
}
KB_DESCRIPTIONS = {
    "core_rules": "Core RPG rules and mechanics.",
    "adversaries": "Monster stats and lore.",
    "environments": "Environment descriptions and hazards.",
    "domain_card_reference": "Description of domain cards and their effects.",
}
# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

chatbox_server = ChatboxServer(DUNGEON_MASTER_API_URL)

@app.route('/knowledge_bases', methods=['GET'])
def get_knowledge_bases():
    kb_list = []
    for kb_name in knowledge_bases.keys():
        kb_list.append({
            "name": kb_name,
            "description": KB_DESCRIPTIONS.get(kb_name, "")
        })
    return jsonify({"knowledge_bases": kb_list})

@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    history = data.get('history', [])
    kb_names = data.get('knowledge_bases', [])
    result, status = chatbox_server.handle_chat(history, kb_names)
    return jsonify(result), status

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})



if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)