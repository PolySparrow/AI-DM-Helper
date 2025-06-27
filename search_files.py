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
CHUNKS_DIR = "./chunks"  # Directory where all .json files are stored
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ========== DYNAMIC LOADING ==========
def load_all_indexes(embeddings_dir, chunks_dir):
    knowledge_bases = {}
    for fname in os.listdir(embeddings_dir):
        if fname.endswith("_embeddings.npy"):
            kb_name = fname.replace("_embeddings.npy", "")
            emb_path = os.path.join(embeddings_dir, fname)
            json_path = os.path.join(chunks_dir, f"{kb_name}_chunks.json")
            if os.path.exists(json_path):
                embeddings = np.load(emb_path)
                with open(json_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                knowledge_bases[kb_name] = {"index": index, "chunks": chunks}
            else:
                print(f"Warning: No chunk file for {kb_name} in {chunks_dir}")
    return knowledge_bases

knowledge_bases = load_all_indexes(EMBEDDINGS_DIR, CHUNKS_DIR)

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
        chunk = chunks[i]
        heading_str = ""
        if chunk.get("headings"):
            heading_str = " | ".join(f"{k}: {v}" for k, v in chunk["headings"].items() if v)
            heading_str = f"[{heading_str}] " if heading_str else ""
        results.append({
            "text": chunk["text"],
            "headings": chunk.get("headings"),
            "score": float(distances[0][rank]),
            "formatted": f"{heading_str}{chunk['text']}"
        })
    print (f"Search results for '{query}' in {kb['index'].ntotal} items: {len(results)} found.")
    return results

def hybrid_search(user_query, k=1, model="gpt-4o"):
    kb_results = {}
    for kb_name, kb in knowledge_bases.items():
        kb_results[kb_name] = search_kb(user_query, kb, k)

    prompt = (
    "You are an expert RPG assistant. Here are possible answers from different knowledge bases:\n\n"
    )
    for kb_name, results in kb_results.items():
        if results:
            for idx, result in enumerate(results):
                section = ""
                if result["headings"]:
                    section = "Section: " + " | ".join(f"{k}: {v}" for k, v in result["headings"].items() if v)
                prompt += (
                    f"{kb_name.capitalize()} (Rank {idx+1}, Score {result['score']:.2f}):\n"
                    f"{section}\n"
                    f"{result['text']}\n"
                )
        else:
            prompt += f"{kb_name.capitalize()}: No relevant info found.\n"
        prompt += "\n"

    prompt += (
        f'The user asked: "{user_query}"\n\n'
        "First, answer the subject of the user's question in your own words, as an expert.\n"
        "Then, pick the most relevant answer from the knowledge bases above, and explain why you chose it.\n"
        "Be sure to state the section heading(s) where you found the answer.\n"
        "If none are relevant, say so."
    )

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
    user_query = "How much hope can a player have?"
    answer = hybrid_search(user_query, k=5)
    print("LLM's answer:\n", answer)