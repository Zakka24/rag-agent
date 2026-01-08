# app/services.py
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.ingestion import Ingestor
from src.pdf_chat import PdfChat
from src.model import Model
from app.config import DATA_FOLDER, STANDARD_PROMPT, DISCLAIMER

class RagService:
    def __init__(self, model: Model, sessions: dict):
        self.model = model
        self.sessions = sessions

    def get_session(self, user_id: str):
        return self.sessions.get(user_id)

    def extract_info_standard(self, user_id: str) -> str:
        session = self.get_session(user_id)
        if not session:
            raise HTTPException(status_code=400, detail="Nessun PDF caricato per questo utente.")

        ingestor: Ingestor = session["ingestor"]
        docs = getattr(ingestor, "documents", [])
        
        if not docs:
            raise HTTPException(status_code=500, detail="Nessun documento indicizzato trovato.")

        # # Ordina documenti
        # docs = sorted(
        #     docs,
        #     key=lambda d: (
        #         d.metadata.get("page") if d.metadata.get("page") is not None else 0,
        #         d.metadata.get("chunk_id", 0),
        #     ),
        # )

        prompt = ChatPromptTemplate.from_messages([
            ("system", STANDARD_PROMPT),
            ("human", (
                "Di seguito trovi il testo (o estratti) del contratto in ordine di pagina.\n"
                "Rispondi solo alle richieste senza aggiungere ulteriori informazioni.\n\n"
                "{context}"
            )),
        ])

        doc_prompt = PromptTemplate.from_template(
            "### PAGINA {page} ({source})\n"
            "{page_content}\n"
            "### FINE PAGINA {page}"
        )

        chain = create_stuff_documents_chain(
            llm=self.model.chat_model,
            prompt=prompt,
            document_prompt=doc_prompt,
            document_separator="\n\n"
        )

        risposta = chain.invoke({"context": docs})
        return risposta + DISCLAIMER

    def process_upload(self, user_id: str, file: UploadFile):
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Il file deve essere un PDF.")

        user_data_folder = DATA_FOLDER / user_id
        user_data_folder.mkdir(parents=True, exist_ok=True)
        dest_path = user_data_folder / file.filename

        session = self.get_session(user_id)

        if dest_path.exists() and session and session.get("file_name") == file.filename:
            answer = self.extract_info_standard(user_id)
            session["standard_info"] = answer
            return {
                "message": "File già presente, uso sessione esistente.",
                "file_already_exists": True,
                "file_name": file.filename,
                "standard_info": answer
            }

        with dest_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        ingestion = Ingestor(
            file_name=file.filename,
            model=self.model,
            user_id=user_id,
        )
        print(f"[{user_id}] Ingestione PDF in corso...")
        ingestion.ingest_file()
        
        chat = PdfChat(model=self.model, ingestor=ingestion)

        self.sessions[user_id] = {
            "file_path": dest_path,
            "file_name": file.filename,
            "ingestor": ingestion,
            "chat": chat,
            "standard_info": None,
        }

        answer = self.extract_info_standard(user_id)
        self.sessions[user_id]["standard_info"] = answer

        return {
            "message": "File PDF caricato e indicizzato con successo.",
            "file_already_exists": False,
            "file_name": file.filename,
            "standard_info": answer
        }

    def ask_question(self, user_id: str, question: str) -> str:
        session = self.get_session(user_id)
        if not session:
            raise HTTPException(status_code=400, detail="Nessun PDF caricato.")
        
        chat: PdfChat = session["chat"]
        answer = chat.ask(question)
        return answer + DISCLAIMER