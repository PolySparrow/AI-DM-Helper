import os
import numpy as np
import json
import faiss
import openai
import config
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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



# ---- 1. Choose and Load a Model ----
# For best results, use a model with "Instruct" or "Chat" in the name.
# See section 2 below for model recommendations.

# Example: Mistral 7B Instruct (needs a good GPU)
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Example: TinyLlama (works on CPU, less capable)


# ---- 2. Format the Prompt ----
def format_prompt(results, user_query):
    prompt = "Here are the top relevant sections for the user's question:\n\n"
    for idx, result in enumerate(results, 1):
        section = ""
        if result.get("headings"):
            section = "Section: " + " | ".join(f"{k}: {v}" for k, v in result["headings"].items() if v)
        prompt += f"{idx}. [{section}]\n{result['text']}\n\n"
    prompt += (
        f'User\'s question: "{user_query}"\n\n'
        "Please answer the user's question using the information above. "
        "Summarize or explain as needed, and cite the section(s) you used."
    )
    return prompt

# ---- 3. Generate the Answer ----
def summarize_with_hf(results, user_query, model, tokenizer, max_new_tokens=300):
    prompt = format_prompt(results, user_query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the output if the model repeats it
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()
    return answer

# ---- 4. Integrate with Your Search ----
def hybrid_search_with_hf(user_query, k=5, model=None, tokenizer=None):
    # You must define knowledge_bases and search_kb elsewhere in your code!
    kb_results = {}
    for kb_name, kb in knowledge_bases.items():
        kb_results[kb_name] = search_kb(user_query, kb, k)

    # Collect all results, flatten, and sort by score
    all_results = []
    for kb_name, results in kb_results.items():
        for result in results:
            result_copy = result.copy()
            result_copy["kb_name"] = kb_name
            all_results.append(result_copy)
    all_results.sort(key=lambda r: r["score"])

    if not all_results:
        return "No relevant information found in any knowledge base."

    top_k = all_results[:k]
    summary = summarize_with_hf(top_k, user_query, model, tokenizer)
    return summary

# ---- 5. Example Usage ----
if __name__ == "__main__":
    model_name = "google/gemma-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    user_query = "How does hope work?"
    # You must define knowledge_bases and search_kb before this!
    answer = hybrid_search_with_hf(user_query, k=2, model=model, tokenizer=tokenizer)
    print(answer)
