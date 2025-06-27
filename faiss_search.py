import numpy as np
import json
import faiss
import openai

# Load embeddings and chunks at module import
EMBEDDINGS_PATH = "sections_embeddings.npy"
CHUNKS_PATH = "sections.json"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Or import from config

# Load data
embeddings = np.load(EMBEDDINGS_PATH)
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    sections = json.load(f)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_query_embedding(query):
    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return np.array([d.embedding for d in response.data], dtype="float32")

def search(query, k=3):
    query_embedding = get_query_embedding(query)
    distances, indices = index.search(query_embedding, k)
    results = []
    for rank, i in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "section": sections[i],
            "distance": float(distances[0][rank])
        })
    return results

# Example usage (remove or comment out in production)
if __name__ == "__main__":
    query = "How do sanity checks work?"
    results = search(query, k=3)
    for r in results:
        print(f"\n--- Section {r['rank']} (distance: {r['distance']:.2f}) ---\n")
        print(r['section'][:1000])