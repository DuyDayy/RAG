import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LVLM (pre-trained vision-language model for generation)
LVLM_MODEL_NAME = "Qwen/Qwen-VL-Chat"

# Multimodal embedding (shared text-image space for retrieval)
EMBEDDING_MODEL = "clip-ViT-B-32"
EMBEDDING_FINETUNED_DIR = "checkpoints/embedding_finetuned"

# Retrieval
TOP_K = 5
MAX_NEW_TOKENS = 512

# Paths
VECTOR_DB_PATH = "data/vector_db/index.faiss"
METADATA_PATH = "data/vector_db/metadata.pkl"
RAW_DATA_DIR = "data/raw/"
FRAMES_CACHE_DIR = "data/cache/frames/"
VIDEOS_DIR = "data/raw/videos/"

# Video
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
VIDEO_SAMPLE_FRAMES = 8       # frame trích khi index / embed
VIDEO_QUERY_FRAMES = 4        # frame đưa vào LVLM mỗi video (tránh OOM)
LVLM_MAX_FRAMES_PER_DOC = 3   # frame tối đa mỗi doc truy xuất đưa vào LVLM

# Agent / RAG prompts
SYSTEM_PROMPT = (
    "Bạn là trợ lý đa phương tiện (ảnh, văn bản, video qua các khung hình). "
    "Sử dụng ảnh/frame ngữ cảnh đã truy xuất và media người dùng "
    "để trả lời chính xác, ngắn gọn, bằng tiếng Việt khi phù hợp."
)