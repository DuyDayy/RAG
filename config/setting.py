import torch

# Configuration settings for the AI Challenge system
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LVLM_MODEL_NAME = "Qwen/Qwen-VL-Chat"
EMBEDDING_MODEL = "clip-ViT-B-32"

# Paths
VECTOR_DB_PATH = "data/vector_db/index.faiss"
METADATA_PATH = "data/vector_db/metadata.pkl"
RAW_DATA_DIR = "data/raw/"