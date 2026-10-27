"""Central configuration for DocMind AI."""
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:0.5b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # approximate words, not model tokens
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "75"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
MAX_DISTANCE = float(os.getenv("RETRIEVAL_MAX_DISTANCE", "1.35"))
REQUEST_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))
