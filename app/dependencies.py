from typing import Dict
from functools import lru_cache
from src.model import Model
import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.config import API_KEY, API_KEY_NAME

user_sessions_db: Dict[str, dict] = {}
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

@lru_cache()
def get_model_instance():
    print("Inizializzazione Modello (vLLM + HF Embeddings)...")

    base_url = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")

    return Model(
        embeddings_model="Qwen/Qwen3-Embedding-0.6B", 
        chat_model='Qwen/Qwen3-4B-Thinking-2507', 
        base_url=base_url
    )

def get_session_store() -> Dict[str, dict]:
    return user_sessions_db

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Credenziali non valide o mancanti."
    )