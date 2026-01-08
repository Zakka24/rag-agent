from fastapi import FastAPI
from app.config import DATA_FOLDER
from app.routers import chat

DATA_FOLDER.mkdir(exist_ok=True)

app = FastAPI(
    title="RAG Qwen PDF Chat",
    version="1.0.0"
)

# Includi il router
app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)