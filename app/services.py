# app/services.py
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document

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

        file_exists = dest_path.exists()

        if not file_exists:
            with dest_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)

            print(f"[{user_id}] File salvato su disco: {file.filename}")
        else:
            print(f"[{user_id}] File già presente. Salto il salvataggio.")

        ingestion = Ingestor(
            file_name=file.filename,
            model=self.model,
            user_id=user_id,
        )

        if not file_exists:
            print(f"[{user_id}] Inizio indicizzazione (ingest_file)...")
            try:
                ingestion.ingest_file()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Errore durante l'ingestione: {str(e)}")
        else:
            print(f"[{user_id}] DB già esistente. Connessione effettuata senza re-ingestione.")
        
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

        message = "File già presente, sessione ripristinata." if file_exists else "File caricato e indicizzato."

        return {
            "message": message,
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