import torch
import multiprocessing
import psutil
from sentence_transformers import SentenceTransformer
import sys
import logging

logger = logging.getLogger(__name__)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_num_cores():
    return multiprocessing.cpu_count()

def get_total_ram_gb():
    return psutil.virtual_memory().total / (1024 ** 3)

def get_gpu_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return 0

def pick_settings():
    device = get_device()
    num_cores = get_num_cores()
    ram_gb = get_total_ram_gb()
    gpu_mem_gb = get_gpu_memory_gb()
    if device == 'cuda':
        if gpu_mem_gb >= 20:
            model = "BAAI/bge-large-en-v1.5"
            batch_size = 128
        elif gpu_mem_gb >= 10:
            model = "BAAI/bge-large-en-v1.5"
            batch_size = 64
        else:
            model = "all-MiniLM-L6-v2"
            batch_size = 32
    else:
        model = "all-MiniLM-L6-v2"
        batch_size = 8 if ram_gb < 8 else 16
    max_workers = min(num_cores, 8)
    return {
        "device": device,
        "model": model,
        "batch_size": batch_size,
        "max_workers": max_workers,
        "ram_gb": ram_gb,
        "gpu_mem_gb": gpu_mem_gb,
    }
logger.info("Python executable:", sys.executable)
logger.info("Torch version:", torch.__version__)
logger.info("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("CUDA device count:", torch.cuda.device_count())
    logger.info("CUDA device name:", torch.cuda.get_device_name(0))
else:
    logger.info("No CUDA device detected.")
settings = pick_settings()
logger.info("Auto-detected settings:", settings)
embedder = SentenceTransformer(settings["model"], device=settings["device"])