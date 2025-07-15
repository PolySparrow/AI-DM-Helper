import os
import numpy as np
import json
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.corpus import wordnet as wn
import dungeon_master_functions as dm_functions
from logging_function import setup_logger
from environment_vars import OLLAMA_URL, OLLAMA_MODEL, EMBEDDING_MODEL, EMBEDDINGS_DIR, CHUNKS_DIR, CROSS_ENCODER_MODEL, MAX_WORKERS,DEVICE
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logger
setup_logger(app_name="AI_DM_RAG")  # or whatever app name you want
logger = logging.getLogger(__name__)

def extract_tags_api(text, top_n=5):
    response = requests.post(
        "http://localhost:5001/extract_tags",  # Your tag extraction endpoint
        json={"text": text, "top_n": top_n}
    )
    response.raise_for_status()
    return response.json().get("tags", [])

def knn_tag_prefilter(chunks, query, k=10, tag_top_n=5):
    logger.debug(f"Prefiltering {len(chunks)} chunks with query: {query}")
    # 1. Extract and embed query tags
    query_tags = extract_tags_api(query, top_n=tag_top_n)
    query_tag_embs = get_embeddings(query_tags)
    logger.debug(f"Query tags: {query_tags}, embeddings shape: {query_tag_embs.shape}")
    # 2. For each chunk, embed its tags and compute max similarity to any query tag
    chunk_scores = []
    for chunk in chunks:
        chunk_tags = chunk.get("tags", [])
        if not chunk_tags:
            continue
        chunk_tag_embs = get_embeddings(chunk_tags)
        # Compute cosine similarity matrix
        sim_matrix = np.dot(query_tag_embs, chunk_tag_embs.T)
        max_sim = np.max(sim_matrix)  # Best match between any query tag and any chunk tag
        chunk_scores.append((chunk, max_sim))
    
    # 3. Sort by similarity and keep top-k
    chunk_scores.sort(key=lambda x: -x[1])
    top_chunks = [c for c, score in chunk_scores[:k]]
    return top_chunks

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
                index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)  # 32 is M, the number of neighbors (tune as needed)
                index.hnsw.efSearch = 64  # (optional) controls recall/speed tradeoff, can tune higher for better recall
                index.add(embeddings)
                knowledge_bases[kb_name] = {"index": index, "chunks": chunks}
            else:
                logger.debug(f"Warning: No chunk file for {kb_name} in {chunks_dir}")
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
def search_kb(query, kb, k=1,prefilter=True, top_n=5):
    index = kb["index"]
    chunks = kb["chunks"]
    if prefilter:
        # Use your prefilter_chunks function
        filtered_chunks = knn_tag_prefilter(chunks, query,k=10 ,tag_top_n=top_n)
        if not filtered_chunks:
            logger.info("No chunks matched pre-filtering, using all chunks.")
            filtered_chunks = chunks
    else:
        filtered_chunks = chunks

    filtered_indices = [chunks.index(chunk) for chunk in filtered_chunks]

    query_embedding = get_embeddings([query])
    assert query_embedding.shape[1] == index.d, f"Embedding dim {query_embedding.shape[1]} != index dim {index.d}"
    distances, indices = index.search(query_embedding, k)

    results = []
    for rank, i in enumerate(indices[0]):
        if i in filtered_indices:
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
            if len(results) >= k:
                break
    return results


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
def get_embeddings(texts):
    response = requests.post("http://localhost:5001/embed", json={"texts": texts})
    return np.array(response.json()["embeddings"])

def rerank(pairs):
    response = requests.post("http://localhost:5001/rerank", json={"pairs": pairs})
    return response.json()["scores"]

def call_llama(prompt, model=OLLAMA_MODEL):
    response = requests.post("http://localhost:5001/llama", json={"model": model, "prompt": prompt, "stream": False})
    return response.json()["response"]

# ========== HYBRID SEARCH: QUERY EXPANSION + RERANKING ==========


def hybrid_search_in_kbs_with_expansion_and_rerank(
    user_query, kb_names, history=None, k=3, model=OLLAMA_MODEL, confidence_threshold=0.5
):
    if isinstance(kb_names, str):
        kb_names = [kb_names]

    # --- 1. Parallel search across all KBs with the original query ---
    def search_one(kb_name):
        if kb_name not in knowledge_bases:
            logger.info(f"Knowledge base '{kb_name}' not found. Skipping.")
            return []
        kb = knowledge_bases[kb_name]
        results = search_kb(user_query, kb, k=10)
        for r in results:
            r["kb_name"] = kb_name
        return results

    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(search_one, kb_name) for kb_name in kb_names]
        for future in as_completed(futures):
            all_results.extend(future.result())

    # Deduplicate by text
    seen = set()
    unique_results = []
    for r in all_results:
        if r["text"] not in seen:
            unique_results.append(r)
            seen.add(r["text"])

    # Rerank
    pairs = [(user_query, r['text']) for r in unique_results]
    if pairs:
        scores = rerank(pairs)
        reranked = sorted(zip(unique_results, scores), key=lambda x: -x[1])
        reranked_results = reranked[:k]
        reranked_results = [(r, score) for r, score in reranked_results if score >= confidence_threshold]
    else:
        reranked_results = []

    # If we have strong matches, return them
    if reranked_results:
        prompt = format_prompt(reranked_results, user_query, history=history)
        summary = call_llama(prompt)
        return summary, list(kb_names)

    # --- 2. If not, do expanded queries in parallel across all KBs ---
    logger.info("No strong matches found, expanding query with WordNet...")
    expanded_queries = expand_query_with_wordnet(user_query)
    all_results = []
    search_args = [(kb_name, q) for kb_name in kb_names for q in expanded_queries]
    def search_expanded(args):
        kb_name, q = args
        if kb_name not in knowledge_bases:
            logger.info(f"Knowledge base '{kb_name}' not found. Skipping.")
            return []
        kb = knowledge_bases[kb_name]
        results = search_kb(q, kb, k=10)
        for r in results:
            r["kb_name"] = kb_name
        return results

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(search_expanded, arg) for arg in search_args]
        for future in as_completed(futures):
            all_results.extend(future.result())

    # Deduplicate again
    seen = set()
    unique_results = []
    for r in all_results:
        if r["text"] not in seen:
            unique_results.append(r)
            seen.add(r["text"])

    # Rerank again
    pairs = [(user_query, r['text']) for r in unique_results]
    if pairs:
        scores = rerank(pairs)
        reranked = sorted(zip(unique_results, scores), key=lambda x: -x[1])
        reranked_results = reranked[:k]
        reranked_results = [(r, score) for r, score in reranked_results if score >= confidence_threshold]
    else:
        reranked_results = []

    # Prepare prompt and summary
    low_confidence_msg = ""
    if not reranked_results:
        low_confidence_msg = (
            "\nWARNING: No high-confidence matches found. "
            "Please try to be more specific in your question.\n"
        )
        logger.info("No reranked results above threshold.")
    prompt = format_prompt(reranked_results, user_query, low_confidence_msg=low_confidence_msg, history=history)
    summary = call_llama(prompt)
    if low_confidence_msg:
        return low_confidence_msg + summary, list(kb_names)
    else:
        return summary, list(kb_names)

# ========== MAIN ==========

if __name__ == "__main__":
    user_query = "When does the DM Generate Fear?"
    kb_name = ["core_rules"]

    logger.info("--- Query Expansion + Reranking Example ---\n")
    answer = hybrid_search_in_kbs_with_expansion_and_rerank(user_query, kb_name, k=5, model=OLLAMA_MODEL)
    logger.info(answer)