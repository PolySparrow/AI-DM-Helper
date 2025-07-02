import os
import numpy as np
import json
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.corpus import wordnet as wn
import dungeon_master_functions as dm_functions
import logging_function

# Setup logger
logger = logging_function.setup_logger()

# ========== CONFIG ==========
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDINGS_DIR = "./embeddings"
CHUNKS_DIR = "./chunks"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

embedder = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder('cross-encoder/ms-marco-electra-base')

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
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                knowledge_bases[kb_name] = {"index": index, "chunks": chunks}
            else:
                print(f"Warning: No chunk file for {kb_name} in {chunks_dir}")
    return knowledge_bases

knowledge_bases = load_all_indexes(EMBEDDINGS_DIR, CHUNKS_DIR)

# ========== QUERY EXPANSION (DYNAMIC SYNONYMS) ==========
def expand_query_with_wordnet(query):
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    words = query.split()
    expanded_queries = set([query])
    for i, word in enumerate(words):
        synsets = wn.synsets(word)
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        for syn in synonyms:
            new_query = " ".join(words[:i] + [syn] + words[i+1:])
            expanded_queries.add(new_query)
    return list(expanded_queries)

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
    return results

# ========== RERANKING ==========
def rerank(query, results, top_n=3):
    pairs = [(query, r['text']) for r in results]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(results, scores), key=lambda x: -x[1])
    return reranked[:top_n]  # List of (result, score) tuples

# ========== PROMPT FORMATTING ==========
def format_prompt(results, user_query, chunk_char_limit=None, low_confidence_msg="", history=None):
    prompt = "You are an expert RPG assistant.\n"
    if low_confidence_msg:
        prompt += low_confidence_msg + "\n"
    for idx, (result, score) in enumerate(results, 1):
        section = ""
        if result.get("headings"):
            section = "Section: " + " | ".join(f"{k}: {v}" for k, v in result["headings"].items() if v)
        chunk_text = result['text'][:chunk_char_limit] if chunk_char_limit else result['text']
        prompt += (
            f"{idx}. [Score: {score:.4f}] [{section}]\n"
            f"{chunk_text}\n\n"
        )
    # Add chat history as context if provided
    if history:
        prompt += "\nConversation so far:\n"
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
        for msg in history:
            role = role_map.get(msg["role"], msg["role"].capitalize())
            content = msg["content"]
            prompt += f"{role}: {content}\n"
    prompt += (
        f'User\'s question: "{user_query}"\n\n'
        "Please answer the user's question using the information above. "
        "Summarize or explain as needed, and cite the section(s) and confidence score(s) you used."
    )
    logger.debug("Formatted prompt for Ollama:\n" + prompt)
    return prompt

# ========== SUMMARIZATION WITH OLLAMA ==========
def summarize_with_ollama(reranked_results, user_query, model=OLLAMA_MODEL, low_confidence_msg="", history=None):
    prompt = format_prompt(reranked_results, user_query, low_confidence_msg=low_confidence_msg, history=history)
    print("\n--- Prompt sent to Ollama ---\n")
    print(prompt)
    print("\n--- End of prompt ---\n")
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# ========== HYBRID SEARCH: QUERY EXPANSION + RERANKING ==========
def hybrid_search_in_kbs_with_expansion_and_rerank(
    user_query, kb_names, history=None, k=3, model=OLLAMA_MODEL, confidence_threshold=0.5
):
    print("I'm a stinky butthole")
    if isinstance(kb_names, str):
        kb_names = [kb_names]
    all_results = []
    expanded_queries = expand_query_with_wordnet(user_query)
    for kb_name in kb_names:
        if kb_name not in knowledge_bases:
            print(f"Knowledge base '{kb_name}' not found. Skipping.")
            continue
        kb = knowledge_bases[kb_name]
        for q in expanded_queries:
            results = search_kb(q, kb, embedder, k=10)
            for r in results:
                r_copy = r.copy()
                r_copy["kb_name"] = kb_name
                all_results.append(r_copy)
    # Deduplicate by text
    seen = set()
    unique_results = []
    for r in all_results:
        if r["text"] not in seen:
            unique_results.append(r)
            seen.add(r["text"])
    # Rerank and get scores
    reranked_results = rerank(user_query, unique_results, top_n=k)

    # Filter out low-confidence results
    reranked_results = [(r, score) for r, score in reranked_results if score >= confidence_threshold]
    # Print confidence scores and chunk info
    print("\nTop reranked results with confidence scores:")
    for idx, (result, score) in enumerate(reranked_results, 1):
        print(f"{idx}. [Score: {score:.4f}] KB: {result.get('kb_name', '')} | Section: {result.get('headings', '')}")
        print(f"   {result['text'][:200]}...\n")
    # Check top score
    top_score = reranked_results[0][1] if reranked_results else 0
    low_confidence_msg = ""
    if not reranked_results:
        low_confidence_msg = (
        "\nWARNING: No high-confidence matches found. "
        "Please try to be more specific in your question.\n"
        )
        logger.info("No reranked results above threshold.")
    else:
        low_confidence_msg = ""
    logger.info(f"Top score: {top_score:.4f} (threshold: {confidence_threshold})")
    logger.debug(f"Reranked results: {reranked_results}")
    # Summarize with Ollama, passing both result and score, and add warning to prompt
    summary = summarize_with_ollama(
        reranked_results, user_query, model=model, low_confidence_msg=low_confidence_msg, history=history
    )
    if low_confidence_msg:
        return low_confidence_msg + summary, list(kb_names)
    else:
        return summary, list(kb_names)

# ========== MAIN ==========

if __name__ == "__main__":
    logger = logging_function.setup_logger()
    user_query = "When does the DM Generate Fear?"
    kb_name = ["core_rules"]

    print("\n--- Query Expansion + Reranking Example ---\n")
    answer = hybrid_search_in_kbs_with_expansion_and_rerank(user_query, kb_name, k=5, model=OLLAMA_MODEL)
    print(answer)