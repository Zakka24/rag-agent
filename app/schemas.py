from pydantic import BaseModel
from typing import Optional, Any

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    reasoning: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    file_already_exists: bool
    file_names: str
    standard_info: str | Any
    status: Optional[str] = None