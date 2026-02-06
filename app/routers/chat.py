# app/routers/chat.py
from fastapi import APIRouter, UploadFile, Request, HTTPException, BackgroundTasks, File, Header, Depends
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.schemas import ChatRequest, ChatResponse, UploadResponse
from app.dependencies import get_model_instance, get_session_store, verify_api_key
from app.services import RagService
from src.model import Model

from typing import List

router = APIRouter(dependencies=[Depends(verify_api_key)])

def get_rag_service(
    model: Model = Depends(get_model_instance),
    sessions: dict = Depends(get_session_store)
) -> RagService:
    return RagService(model, sessions)

@router.post("/upload_pdf", response_model=UploadResponse)
async def upload_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(...),
    service: RagService = Depends(get_rag_service)
):
    form = await request.form()
    uploaded = []

    for k, v in form.multi_items():
        if isinstance(v, StarletteUploadFile) and (k == "files" or k == "files[]" or k.startswith("files[")):
            uploaded.append(v)
        
        if not uploaded:
            raise HTTPException(status_code=422, detail=[{"loc": ["body", "files"], "msg": "Field required ttt", "type": "missing"}])
            
    return service.process_upload(x_user_id, uploaded, background_tasks)

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest, 
    x_user_id: str = Header(...),
    service: RagService = Depends(get_rag_service)
):
    result = service.ask_question(x_user_id, req.question)
    return ChatResponse(
        answer=result["answer"],
        reasoning=result["reasoning"]
    )

@router.get("/status")
async def check_status(
    x_user_id: str = Header(...),
    service: RagService = Depends(get_rag_service)
):
    return service.get_analysis_status(x_user_id)