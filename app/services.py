# app/services.py
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document

from typing import List

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
            print(f"[{user_id}] Documenti non in memoria. Recupero dal Vector Store...")
            db_data = ingestor.vector_store.get()
            if db_data and db_data['documents']:
                for text, metadata in zip(db_data['documents'], db_data['metadatas']):
                    docs.append(Document(page_content=text, metadata=metadata))

            docs = sorted(
                docs, key=lambda d: d.metadata.get("chunk_index", 0)
            )

        if not docs:
            raise HTTPException(status_code=500, detail="Nessun documento indicizzato trovato.")

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
        try:
            risposta = chain.invoke({"context": docs})
        except Exception as e:
            print(f"[{user_id}] Errore imprevisto LLM: {e}")
            raise HTTPException(status_code=500, detail=f"Errore nella generazione della risposta: {str(e)}")
        
        return risposta + DISCLAIMER

    def process_upload(self, user_id: str, files: List[UploadFile]):
        user_data_folder = DATA_FOLDER / user_id
        user_data_folder.mkdir(parents=True, exist_ok=True)

        ingestor = Ingestor(model=self.model, user_id=user_id)

        saved_paths = []
        filenames = []

        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue
            
            dest_path = user_data_folder / file.filename
            filenames.append(file.filename)
            saved_paths.append(dest_path)
            
            with dest_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            print(f"[{user_id}] File salvato: {file.filename}")

        if not saved_paths:
            raise HTTPException(status_code=400, detail="Nessun PDF valido fornito.")

        print(f"[{user_id}] Ingestione di {len(saved_paths)} nuovi file...")
        try:
            ingestor.ingest_files(saved_paths)
        except Exception as e:
            shutil.rmtree(user_data_folder)
            raise HTTPException(status_code=500, detail=f"Errore ingestione: {str(e)}")

        chat = PdfChat(model=self.model, ingestor=ingestor)
        self.sessions[user_id] = {
            "file_paths": saved_paths,
            "file_names": filenames,
            "ingestor": ingestor,
            "chat": chat,
            "standard_info": None
        }

        answer = self.extract_info_standard(user_id)
        self.sessions[user_id]["standard_info"] = answer

        return {
            "message": "Nuova analisi avviata. Contesto aggiornato.",
            "file_names": ", ".join(filenames),
            "standard_info": answer,
            "file_already_exists": False 
        }

    def ask_question(self, user_id: str, question: str) -> str:
        session = self.get_session(user_id)
        if not session:
            raise HTTPException(status_code=400, detail="Nessun PDF caricato.")
        
        chat: PdfChat = session["chat"]
        try:
            answer = chat.ask(question)
        except Exception as e:
            print(f"[{user_id}] Errore imprevisto LLM: {e}")
            raise HTTPException(status_code=500, detail=f"Errore nella generazione della risposta: {str(e.message)}")
        
        return answer + DISCLAIMER