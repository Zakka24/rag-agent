from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from src.ingestion import Ingestor
from src.model import Model
from typing import List, Tuple, Optional
import re


class PdfChat:
    """
    Class to handle the interactive chat
    """

    def __init__(self, model: Model, ingestor: Ingestor, prompt_message: Optional[List[Tuple[str]]] = None):
        """
        Initialize the PdfChat instance with the model, ingestor, and an optional custom prompt message.

        Immediately defines the chat prompt and sets up the retrieval chain for
        generating responses based on the ingested PDF data.

        Args:
            model (Model): The model used for generating responses.
            ingestor (Ingestor): The Ingestor object responsible for managing the vector store.
            prompt_message (Optional[List[Tuple[str]]]): A custom prompt message to guide the assistant's responses.
        """

        self.model = model
        self.ingestor = ingestor
        self.prompt_message = prompt_message
        self._define_prompt()
        self._define_retrieval_chain()

    def _define_prompt(self):
        """
        Defines the system and user prompt messages to guide the assistant's behavior during the chat
        """

        if not self.prompt_message:
            prompt_message = [
                ('system', 
                        "Sei un assistente legale esperto e rigoroso. Rispondi SEMPRE in italiano. \n"
                        "Il tuo compito è rispondere alle domande basandoti **ESCLUSIVAMENTE** sul contesto fornito.\n\n"

                        "**STRUTTURA DEI DOCUMENTI**\n"
                        "Il contesto può contenere frammenti da DIVERSI documenti (es. Contratto Preliminare, Atti Integrativi, Certificati di Morte, Visure).\n"
                        "Presta estrema attenzione all'intestazione `[FILE: ...]` presente nel testo per distinguere le fonti.\n\n"

                        "**REGOLE FONDAMENTALI DI RISOLUZIONE DEI CONFLITTI**\n"
                        "1. **Cronologia:** I documenti successivi modificano o annullano quelli precedenti. (Es. Un 'Atto di Decesso' o una 'Cessione' successiva al contratto cambia il beneficiario).\n"
                        "2. **Prevalenza:** Se il contratto dice 'Pagare a Mario' ma un documento successivo dice 'Mario è deceduto, pagare agli eredi', la risposta corretta è **gli eredi**.\n"
                        "3. **Completezza:** Se c'è stata una variazione, spiegalo. (Es. 'Il beneficiario originale era X, ma in seguito al decesso riportato nel file Y, il nuovo beneficiario è Z').\n\n"

                        "**GESTIONE SINONIMI LEGALI**\n"
                        "Considera equivalenti termini come:\n"
                        "- Parte Promittente Venditrice / Venditore / Dante causa\n"
                        "- Parte Promittente Acquirente / Acquirente / Avente causa\n"
                        "- Corrispettivo / Prezzo / Canone\n\n"

                        "**ISTRUZIONI DI RISPOSTA**\n"
                        "SE trovi la risposta:\n"
                        "- Estraila dal documento più aggiornato/rilevante.\n"
                        "- Cita SEMPRE la fonte: 'Secondo il file [NOME FILE] a pagina [X]...'.\n"
                        "- Se la situazione è evoluta nel tempo, ricostruisci la storia: 'Inizialmente... successivamente...'.\n\n"

                        "SE la risposta NON è nel contesto:\n"
                        "- Dì chiaramente: “Nel contesto fornito non trovo questa informazione.”\n"
                        "- Non inventare MAI. Non fare deduzioni non supportate dal testo.\n"
                ),
                ('human', 
                    "Domanda dell’utente: {input}\n\n"

                    "Contesto disponibile (può contenere più file):\n"
                    "{context}\n\n"

                    "Rispondi alla domanda tenendo conto di eventuali modifiche intervenute tra i documenti."
                )
            ]

            self.prompt_message = prompt_message

        self.prompt = ChatPromptTemplate.from_messages(self.prompt_message)

    def _define_retrieval_chain(self):
        self.retriever = self.ingestor.vector_store.as_retriever(
            search_kwargs={"k": 50}
        )

        combine_docs_chain = create_stuff_documents_chain(
            self.model.chat_model,
            self.prompt
        )

        self.retrieval_chain = create_retrieval_chain(
            self.retriever,
            combine_docs_chain
        )

    def ask(self, query: str) -> dict:
        """
        Asks one single question to LLM and return the response.
        """

        result = self.retrieval_chain.invoke({"input": query})
        full_text = result["answer"]

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