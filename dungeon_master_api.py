from flask import Flask, request, jsonify, render_template


from datetime import datetime
import logging_function
import numpy as np
import dungeon_master_functions as dm_functions
import search_files as search_files
import requests
from build_embeddings import embedding_generator
from sentence_transformers import SentenceTransformer
import os
# Setup logger
logger = logging_function.setup_logger()
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"


app = Flask(__name__)

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
embedder = SentenceTransformer(EMBEDDING_MODEL)

@app.route('/')
def home():
    """Return the home page."""
    return render_template('~Endpoints~<br/>' 
    '/api/v1.0/modeler_agent<br/>')
    

def get_embeddings(texts, client):
    response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small"
    )
    return np.array([d.embedding for d in response.data], dtype="float32")


#@app.route('/api/v1.0/chat', methods=['POST'])
#def CallChat():
#    f = json.loads(request.data)
#    response = dm_functions.Chat(f)
#    return response

@app.route('/api/v1.0/roll_dice', methods=['POST'])
def roll_dice_route():
    data = request.get_json()
    dice_notation = data.get("dice", "")
    result=dm_functions.roll_dice(dice_notation)

@app.route('/api/v1.0/hybrid_search', methods=['POST'])
def hybrid_search_route():
    logger.debug("Received request for hybrid search")
    data = request.get_json()
    history = data.get("history", [])
    kb_names = data.get("knowledge_bases", [])
    logger.debug(f"History: {history}")
    logger.debug(f"Knowledge Bases: {kb_names}")    
    user_query = dm_functions.get_latest_user_message(history)
    logger.debug(f"User query: {user_query}")
    history = dm_functions.build_hybrid_history(history, model=OLLAMA_MODEL)
    if not user_query:
        return jsonify({"success": False, "error": "No user message found in history"}), 400

    try:
        answer, used_kbs = search_files.hybrid_search_in_kbs_with_expansion_and_rerank(
            user_query, kb_names, history=history, k=3, model=OLLAMA_MODEL
        )
        logger.debug(f"Answer: {answer}")
        logger.debug(f"Used Knowledge Bases: {used_kbs}")

        return jsonify({
            "success": True,
            "response": answer,
            "used_kbs": used_kbs
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/refresh_embeddings', methods=['POST'])
def refresh_embeddings():
    data = request.get_json()
    kb_names = data.get('kb_names', [])
    if not kb_names:
        return jsonify({'success': False, 'error': 'No KB names provided'}), 400

    results = {}
    for kb_name in kb_names:
        file_path = dm_functions.find_kb_file(kb_name)
        if not file_path:
            results[kb_name] = "File not found"
            continue
        try:
            embedding_generator(
                FILE_PATH=file_path,
                embedder=embedder,
                BATCH_SIZE=20,
                headings_csv_path="./source/Daggerheart_context_extended.csv"
            )
            results[kb_name] = "Success"
        except Exception as e:
            results[kb_name] = f"Exception: {str(e)}"
    return jsonify({'success': True, 'results': results})

@app.route('/knowledge_bases', methods=['GET'])
def get_knowledge_bases():
    source_dir = "./source"
    KB_DESCRIPTIONS = {
        "core_rules": "Core RPG rules and mechanics.",
        "adversaries": "Monster stats and lore.",
        "environments": "Environment descriptions and hazards.",
        "domain_card_reference": "Description of domain cards and their effects.",
    }
    kb_list = []
    for fname in os.listdir(source_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() in [".txt", ".pdf", ".docx", ".csv", ".xlsx"]:
            kb_list.append({
                "name": base,
                "description": KB_DESCRIPTIONS.get(base, "")
            })
    return jsonify({"knowledge_bases": kb_list})

if __name__ == "__main__":
    logger = logging_function.setup_logger()
    logger.debug("Starting the Dungeon Master API Flask app...")
    # Start the Flask app
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(debug=True,host='127.0.0.1', port=5001, use_reloader=False)

