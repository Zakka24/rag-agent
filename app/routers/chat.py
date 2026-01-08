# app/routers/chat.py
from fastapi import APIRouter, UploadFile, File, Header, Depends
from app.schemas import ChatRequest, ChatResponse, UploadResponse
from app.dependencies import get_model_instance, get_session_store
from app.services import RagService
from src.model import Model

router = APIRouter()

def get_rag_service(
    model: Model = Depends(get_model_instance),
    sessions: dict = Depends(get_session_store)
) -> RagService:
    return RagService(model, sessions)

@router.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...), 
    x_user_id: str = Header(...),
    service: RagService = Depends(get_rag_service)
):
    return service.process_upload(x_user_id, file)

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest, 
    x_user_id: str = Header(...),
    service: RagService = Depends(get_rag_service)
):
    answer = service.ask_question(x_user_id, req.question)
    return ChatResponse(answer=answer)