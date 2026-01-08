# from typing import Dict
# from functools import lru_cache
# from src.model import Model
# from app.config import OLLAMA_URL

# user_sessions_db: Dict[str, dict] = {}

# @lru_cache()
# def get_model_instance():
#     """
#     Inizializza il modello una volta sola e lo riutilizza (Singleton).
#     """
#     print("Inizializzazione Modello LLM...")
#     return Model(
#         # embeddings_model='embeddinggemma:300m',
#         embeddings_model='qwen3-embedding:8b',
#         chat_model='qwen3:14b',
#         ollama_host=OLLAMA_URL
#     )

# def get_session_store() -> Dict[str, dict]:
#     return user_sessions_db

from typing import Dict
from functools import lru_cache
from src.model import Model
import os
from pathlib import Path

user_sessions_db: Dict[str, dict] = {}

@lru_cache()
def get_model_instance():
    print("Inizializzazione Modello LLM...")

    local_model_dir = os.getenv("CHAT_MODEL_PATH")
    if not local_model_dir:
        base_dir = Path(__file__).resolve().parents[1]
        # local_model_dir = str(base_dir / "models" / "Qwen3-14B-local")
        local_model_dir = str(base_dir / "models" / "Qwen3-4B-Thinking-2507")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    # print(local_model_dir)

    return Model(
        # embeddings_model="Qwen/Qwen3-Embedding-8B",
        embeddings_model="qwen3-embedding:0.6b",
        # chat_model="Qwen/Qwen3-14B",
        # chat_model=local_model_dir,
        # chat_model='ServiceNow-AI/Apriel-1.6-15b-Thinker:Q4_K_M',
        chat_model='qwen3-vl:8b',
        base_url=base_url
    )

def get_session_store() -> Dict[str, dict]:
    return user_sessions_db
