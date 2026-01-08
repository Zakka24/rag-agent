from pydantic import BaseModel
from typing import Optional, Any

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

class UploadResponse(BaseModel):
    message: str
    file_already_exists: bool
    file_name: str
    standard_info: str | Any