# app/services.py
import shutil
from pathlib import Path
from fastapi import UploadFile, HTTPException, BackgroundTasks
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.ingestion import Ingestor
from src.pdf_chat import PdfChat
from src.model import Model
from app.config import DATA_FOLDER, STANDARD_PROMPT, DISCLAIMER, MAP_PROMPT_TEXT, MAX_GROUP_CHARS

from typing import List
import re

class RagService:
    def __init__(self, model: Model, sessions: dict):
        self.model = model
        self.sessions = sessions

    def get_session(self, user_id: str):
        return self.sessions.get(user_id)
    
    def get_analysis_status(self, user_id: str):
        session = self.sessions.get(user_id)
        if not session:
            return {"status": "not_found"}
        
        return {
            "status": session.get("status", "processing"),
            "data": session.get("standard_info"),
            "error": session.get("error_message")
        }
    
    def get_chat_status(self, user_id: str):
        session = self.get_session(user_id)
        if not session:
            return {"status": "not_found"}
        
        return {
            "status": session.get("chat_status", "idle"),
            "result": session.get("chat_last_result"),
            "error": session.get("chat_error")
        }
    
    def run_async_analysis(self, user_id: str):
        try:
            print(f"[{user_id}] Avvio analisi asincrona...")
            answer = self.extract_info_standard(user_id)
            
            if user_id in self.sessions:
                self.sessions[user_id]["standard_info"] = answer
                self.sessions[user_id]["status"] = "completed"
                print(f"[{user_id}] Analisi completata con successo.")
                
        except Exception as e:
            print(f"[{user_id}] Errore background: {e}")
            if user_id in self.sessions:
                self.sessions[user_id]["status"] = "error"
                self.sessions[user_id]["error_message"] = str(e)

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
        
        grouped_docs = []
        current_group = []
        current_char_count = 0

        for doc in docs:
            doc_len = len(doc.page_content)
            if current_char_count + doc_len > MAX_GROUP_CHARS:
                grouped_docs.append(current_group)
                current_group = [doc]
                current_char_count = doc_len
            else:
                current_group.append(doc)
                current_char_count += doc_len
        if current_group:
            grouped_docs.append(current_group)

        print(f"[{user_id}] Diviso in {len(grouped_docs)} gruppi di analisi.")

        map_prompt = ChatPromptTemplate.from_template(MAP_PROMPT_TEXT)
        map_chain = map_prompt | self.model.chat_model | StrOutputParser()
        
        intermediate_results = []
        
        for i, group in enumerate(grouped_docs):
            group_text = "\n\n".join([f"--- Pagina {d.metadata.get('page', '?')} ---\n{d.page_content}" for d in group])
            
            print(f"[{user_id}] Analisi gruppo {i+1}/{len(grouped_docs)}...")
            try:
                res = map_chain.invoke({"context": group_text})
                full_text = res
                separator = "</think>"
                split_index = full_text.rfind(separator)
                
                if split_index != -1: 
                    raw_thinking = full_text[:split_index]
                    thinking_content = raw_thinking.replace("<think>", "").strip()
                    content = full_text[split_index + len(separator):].strip()
                    print('think found')
                else:
                    thinking_content = ""
                    content = full_text.strip()
                    print('no think found')

                intermediate_results.append(content)

            except Exception as e:
                print(f"Errore nel batch {i}: {e}")
                continue

        print(f"[{user_id}] Sintesi finale dei dati estratti...")
        
        full_extracted_context = "\n\n".join(intermediate_results)

        print(full_extracted_context)

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", STANDARD_PROMPT),
            ("human", (
                "Qui di seguito trovi gli appunti estratti dall'analisi sequenziale del documento.\n"
                "Usa SOLO queste informazioni per compilare le tabelle finali richieste.\n"
                "Se noti discrepanze (es. pagina 1 dice durata X, pagina 100 dice Y), riportale.\n\n"
                "DATI ESTRATTI:\n{context}"
            )),
        ])

        final_chain = final_prompt | self.model.chat_model | StrOutputParser()
        risposta_finale = final_chain.invoke({"context": full_extracted_context})
        full_text = risposta_finale

        separator = "</think>"
        split_index = full_text.rfind(separator)
        
        if split_index != -1:
            raw_thinking = full_text[:split_index]
            thinking_content = raw_thinking.replace("<think>", "").strip()
            content = full_text[split_index + len(separator):].strip()
        else:
            thinking_content = ""
            content = full_text.strip()        
        return {
            "answer": content,
            "reasoning": thinking_content
        }

    def process_upload(self, user_id: str, files: List[UploadFile], background_tasks : BackgroundTasks):
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
            "standard_info": None,
            "status": "processing"
        }

        background_tasks.add_task(self.run_async_analysis, user_id)

        return {
            "message": "Upload ricevuto. Analisi avviata in background.",
            "file_names": ", ".join(filenames),
            "standard_info": None,
            "file_already_exists": False,
            "status": "processing"
        }
    
    def run_background_chat(self, user_id: str, question: str):
        try:
            session = self.get_session(user_id)
            if not session:
                return

            print(f"[{user_id}] Elaborazione domanda chat in background...")
            chat: PdfChat = session["chat"]
            response = chat.ask(question)
            
            response["answer"] = response["answer"] + DISCLAIMER
            
            session["chat_status"] = "completed"
            session["chat_last_result"] = response
            print(f"[{user_id}] Risposta chat pronta.")

        except Exception as e:
            print(f"[{user_id}] Errore chat background: {e}")
            if user_id in self.sessions:
                self.sessions[user_id]["chat_status"] = "error"
                self.sessions[user_id]["chat_error"] = str(e)

    def ask_question(self, user_id: str, question: str, background_tasks: BackgroundTasks) -> dict:
        session = self.get_session(user_id)
        if not session:
            raise HTTPException(status_code=400, detail="Nessun PDF caricato.")
        
        session["chat_status"] = "processing"
        session["chat_last_result"] = None
        session["chat_error"] = None

        print(f"[{user_id}] Question: {question}")


        background_tasks.add_task(self.run_background_chat, user_id, question)

        return {
            "status": "processing",
            "answer": None,
            "reasoning": None
        }