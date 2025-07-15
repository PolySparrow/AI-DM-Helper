import os
import logging
from hardware_adjustment_scripts import pick_settings
settings = pick_settings()
# settings = {
#     "device": "cuda" or "cpu",
#     "model": "BAAI/bge-large-en-v1.5" or "all-MiniLM-L6-v2",
#     "batch_size": 128, 64, 32, etc.
#     "max_workers": 8, etc.
#     "ram_gb": ...,
#     "gpu_mem_gb": ...,
# }
# LLM/Embedding Models


DEVICE = settings["device"]
OLLAMA_URL = "http://localhost:11434/api/generate"
#OLLAMA_MODEL = "phi-3-mini-128k-instruct.Q4_K_M.gguf" # Adjust as needed
OLLAMA_MODEL = "phi3"  # Adjust as needed
EMBEDDING_MODEL = settings["model"]
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-electra-base"

DM_API_URL = "http://127.0.0.1:5001/api/v1.0/hybrid_search"

BATCH_SIZE = settings["batch_size"]
MAX_WORKERS = settings["max_workers"]

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "source")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")


# Logging
LOG_FILE = "./AI_DM.log"
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