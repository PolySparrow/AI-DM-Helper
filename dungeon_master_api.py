from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from logging_function import setup_logger
import numpy as np
import dungeon_master_functions as dm_functions
import search_files as search_files
import requests
from build_embeddings import embedding_generator
from sentence_transformers import SentenceTransformer
from environment_vars import 	OLLAMA_URL, OLLAMA_MODEL, EMBEDDING_MODEL, SOURCE_DIR, KNOWLEDGE_BASES, KB_DESCRIPTIONS, MAX_WORKERS
import logging

import os
# Setup logger




app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
setup_logger(app_name="AI_DM_RAG")  # or whatever app name you want
logger = logging.getLogger(__name__)

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
    logger.info("Received request to /refresh_embeddings")
    data = request.get_json()
    logger.debug(f"Request JSON: {data}")

    kb_names = data.get('kb_names', [])
    if not kb_names:
        logger.warning("No KB names provided in request")
        return jsonify({'success': False, 'error': 'No KB names provided'}), 400

    results = {}

    def process_kb(kb_name):
        file_path = dm_functions.find_kb_file(kb_name)
        if not file_path:
            logger.error(f"File not found for KB: {kb_name}")
            return (kb_name, "File not found")
        try:
            logger.info(f"Starting embedding generation for {kb_name} at {file_path}")
            embedding_generator(
                FILE_PATH=file_path,
                embedder=embedder,
                BATCH_SIZE=128,
                headings_csv_path="./source/Daggerheart_context_extended.csv"
            )
            logger.info(f"Successfully refreshed embeddings for {kb_name}")
            return (kb_name, "Success")
        except Exception as e:
            logger.exception(f"Exception while refreshing {kb_name}: {e}")
            return (kb_name, f"Exception: {str(e)}")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(kb_names))) as executor:
        # Submit all jobs
        future_to_kb = {executor.submit(process_kb, kb_name): kb_name for kb_name in kb_names}
        for future in as_completed(future_to_kb):
            kb_name, result = future.result()
            results[kb_name] = result

    logger.info(f"Refresh results: {results}")
    return jsonify({'success': True, "response": 'Knowledge Base Refreshed', 'results': results})

@app.route('/knowledge_bases', methods=['GET'])
def get_knowledge_bases():
    source_dir = SOURCE_DIR
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
    logger.debug("Starting the Dungeon Master API Flask app...")
    # Start the Flask app
    for rule in app.url_map.iter_rules():
        logger.debug(rule)
    app.run(debug=True,host='127.0.0.1', port=5001, use_reloader=False)

