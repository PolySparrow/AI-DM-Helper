import os
import logging

# LLM/Embedding Models
OLLAMA_URL = "http://localhost:11434/api/generate"
#OLLAMA_MODEL = "phi-3-mini-128k-instruct.Q4_K_M.gguf" # Adjust as needed
OLLAMA_MODEL = "llama3"  # Adjust as needed
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # or "bge-base-en-v1.5" if you want
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-electra-base"
#CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Adjust as needed
DM_API_URL = "http://127.0.0.1:5001/api/v1.0/hybrid_search"

BATCH_SIZE = 128  # Adjust based on your GPU memory
MAX_WORKERS = 8  # Adjust based on your CPU cores

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "source")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")

# Logging
LOG_FILE = "AI_DM.log"
LOG_LEVEL = logging.DEBUG

# Knowledge Bases
KNOWLEDGE_BASES = {
    "core_rules": {},
    "adversaries": {},
    "environments": {},
    "domain_card_reference": {},
}
KB_DESCRIPTIONS = {
    "core_rules": "Core RPG rules and mechanics.",
    "adversaries": "Monster stats and lore.",
    "environments": "Environment descriptions and hazards.",
    "domain_card_reference": "Description of domain cards and their effects.",
}