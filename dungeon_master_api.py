from flask import Flask, request, jsonify, render_template
import json
import openai
import config
from datetime import datetime
import logging_function
import numpy as np
import faiss_search
import dungeon_master_functions as dm_functions
import search_files

# Setup logger
logger = logging_function.setup_logger()



app = Flask(__name__)

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

@app.route('/api/v1.0/search', methods=['POST'])
def search_route():
    data = request.get_json()
    query = data.get("query", "")
    results=search_files.hybrid_search(query, k=5)
    return jsonify({"results": results})

@app.route('/api/v1.0/chat', methods=['POST'])
def CallChat():
    f = json.loads(request.data)
    response = dm_functions.Chat(f)
    return response

if __name__ == "__main__":
    logger.debug("Starting the Dungeon Master API Flask app...")
    # Start the Flask app
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(debug=True,host='127.0.0.1', port=5001, use_reloader=False)
