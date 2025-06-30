import os
import numpy as np
import json
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ========== CONFIG ==========
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDINGS_DIR = "./embeddings"
CHUNKS_DIR = "./chunks"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # or "llama3:70b" for 70B, or "mistral", etc.

# ========== EMBEDDER ==========
embedder = SentenceTransformer(EMBEDDING_MODEL)

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
                index = faiss.IndexFlatIP(embeddings.shape[1])  # Use inner product for normalized embeddings
                index.add(embeddings)
                knowledge_bases[kb_name] = {"index": index, "chunks": chunks}
            else:
                print(f"Warning: No chunk file for {kb_name} in {chunks_dir}")
    return knowledge_bases

knowledge_bases = load_all_indexes(EMBEDDINGS_DIR, CHUNKS_DIR)

# ========== DYNAMIC SEARCH ==========
def search_kb(query, kb, embedder, k=1):
    index = kb["index"]
    chunks = kb["chunks"]
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    assert query_embedding.shape[1] == index.d, f"Embedding dim {query_embedding.shape[1]} != index dim {index.d}"
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
    print(f"Search results for '{query}' in {kb['index'].ntotal} items: {len(results)} found.")
    return results

# ========== PROMPT FORMATTING ==========
def format_prompt(results, user_query, chunk_char_limit=None):
    prompt = "You are an expert RPG assistant.\n"
    for idx, result in enumerate(results, 1):
        section = ""
        if result.get("headings"):
            section = "Section: " + " | ".join(f"{k}: {v}" for k, v in result["headings"].items() if v)
        chunk_text = result['text'][:chunk_char_limit]
        prompt += f"{idx}. [{section}]\n{chunk_text}\n\n"
    prompt += (
        f'User\'s question: "{user_query}"\n\n'
        "Please answer the user's question using the information above. "
        "Summarize or explain as needed, and cite the section(s) you used."
    )
    return prompt

# ========== OLLAMA SUMMARIZATION ==========
def summarize_with_ollama(results, user_query, model=OLLAMA_MODEL):
    prompt = format_prompt(results, user_query)
    print("\n--- Prompt sent to Ollama ---\n")
    print(prompt)
    print("\n--- End of prompt ---\n")
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# ========== HYBRID SEARCH + SUMMARIZATION ==========
def hybrid_search_with_ollama(user_query, k=3, model=OLLAMA_MODEL):
    kb_results = {}
    for kb_name, kb in knowledge_bases.items():
        kb_results[kb_name] = search_kb(user_query, kb, embedder, k)

    # Collect all results, flatten, and sort by score (higher is better for cosine similarity)
    all_results = []
    for kb_name, results in kb_results.items():
        for result in results:
            result_copy = result.copy()
            result_copy["kb_name"] = kb_name
            all_results.append(result_copy)
    all_results.sort(key=lambda r: -r["score"])  # Sort descending

    if not all_results:
        return "No relevant information found in any knowledge base."

    top_k = all_results[:k]
    summary = summarize_with_ollama(top_k, user_query, model=model)
    return summary

def hybrid_search_in_kb(user_query, kb_name, k=3, model=OLLAMA_MODEL):
    if kb_name not in knowledge_bases:
        return f"Knowledge base '{kb_name}' not found."
    kb = knowledge_bases[kb_name]
    results = search_kb(user_query, kb, embedder, k)
    if not results:
        return f"No relevant information found in knowledge base '{kb_name}'."
    summary = summarize_with_ollama(results[:k], user_query, model=model)
    return summary

# ========== MAIN ==========
if __name__ == "__main__":
    user_query = "How do players generate hope?"
    #answer = hybrid_search_with_ollama(user_query, k=5, model=OLLAMA_MODEL)
    #print("\n--- LLM's answer ---\n")
    #print(answer)
    #print("\n--- End of answer ---\n")
    # Example usage for a specific knowledge base
    kb_name = "core_rules"
    answer = hybrid_search_in_kb(user_query, kb_name, k=5, model=OLLAMA_MODEL)
    print(f"\n--- LLM's answer from {kb_name} ---\n")
    print(answer)
