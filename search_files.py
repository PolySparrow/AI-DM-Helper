import os
import numpy as np
import json
import faiss
import openai
import config

# ========== CONFIG ==========
OPENAI_API_KEY = config.openai_apikey
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDINGS_DIR = "./embeddings"  # Directory where all .npy/.json files are stored

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ========== DYNAMIC LOADING ==========
def load_all_indexes(embeddings_dir):
    knowledge_bases = {}
    for fname in os.listdir(embeddings_dir):
        if fname.endswith("_embeddings.npy"):
            kb_name = fname.replace("_embeddings.npy", "")
            emb_path = os.path.join(embeddings_dir, fname)
            json_path = os.path.join(embeddings_dir, f"{kb_name}.json")
            if os.path.exists(json_path):
                embeddings = np.load(emb_path)
                with open(json_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                knowledge_bases[kb_name] = {"index": index, "chunks": chunks}
    return knowledge_bases

knowledge_bases = load_all_indexes(EMBEDDINGS_DIR)

# ========== EMBEDDING FUNCTION ==========
def get_query_embedding(query):
    response = openai_client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return np.array([d.embedding for d in response.data], dtype="float32")

# ========== DYNAMIC SEARCH ==========
def search_kb(query, kb, k=1):
    index = kb["index"]
    chunks = kb["chunks"]
    query_embedding = get_query_embedding(query)
    distances, indices = index.search(query_embedding, k)
    results = []
    for rank, i in enumerate(indices[0]):
        results.append(chunks[i])
    return results

def hybrid_search(user_query, k=1, model="gpt-4o"):
    # 1. Search all knowledge bases
    kb_results = {}
    for kb_name, kb in knowledge_bases.items():
        kb_results[kb_name] = search_kb(user_query, kb, k)

    # 2. Build prompt dynamically
    prompt = "You are an expert RPG assistant. Here are possible answers from different knowledge bases:\n\n"
    for kb_name, results in kb_results.items():
        result_text = results[0] if results else f"No relevant info found in {kb_name}."
        prompt += f"{kb_name.capitalize()}: {result_text}\n\n"
    prompt += (
        f'Based on the user\'s question: "{user_query}", pick the most relevant answer and explain why. '
        "If none are relevant, say so."
    )

    # 3. Call the LLM
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful RPG rules assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=512,
        temperature=0.2
    )
    return response.choices[0].message.content

# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    user_query = "How do sanity checks work?"
    answer = hybrid_search(user_query, k=1)
    print("LLM's answer:\n", answer)