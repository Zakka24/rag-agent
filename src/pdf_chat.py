from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from src.ingestion import Ingestor
from src.model import Model
from typing import List, Tuple, Optional


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
                        "Sei un assistente legale che risponde SEMPRE in italiano e basandoti "
                        "ESCLUSIVAMENTE sul contesto fornito. Non assumere nulla che non sia "
                        "presente nel contesto. \n\n"

                        "SE trovi la risposta:\n"
                        "- estraila ESATTAMENTE dal contesto (mai inventare)\n"
                        "- riassumila in modo chiaro\n"
                        "- specifica in quale parte del contesto è stata trovata (citazione breve)\n"
                        "- forniscimi la pagina del documento dove l'hai trovata\n"
                        "- se nel documento sembrano esserci più risposte alla domanda fatta, forniscile tutte quante.\n"

                        "SE la risposta NON è nel contesto:\n"
                        "- dì chiaramente: “Nel contesto fornito non trovo questa informazione.\n”"

                        "SE la domanda dell’utente è vaga, richiedi chiarimenti.\n"
                        "Non usare conoscenze esterne. Non fare deduzioni, non completare parti mancanti.\n"
                ),
                ('human', 
                    "Domanda dell’utente: {input}"

                    "Contesto disponibile: "
                    "{context}"

                    "Rispondi basandoti SOLO sul contesto. Se utile, cita espressamente la parte "
                    "del testo da cui hai ricavato la risposta."
                )
            ]

            self.prompt_message = prompt_message

        self.prompt = ChatPromptTemplate.from_messages(self.prompt_message)

    def _define_retrieval_chain(self):
        self.retriever = self.ingestor.vector_store.as_retriever(
            search_kwargs={"k": 20}
        )

        combine_docs_chain = create_stuff_documents_chain(
            self.model.chat_model,
            self.prompt
        )

        self.retrieval_chain = create_retrieval_chain(
            self.retriever,
            combine_docs_chain
        )

    def ask(self, query: str) -> str:
        """
        Asks one single question to LLM and return the response.
        """

        result = self.retrieval_chain.invoke({"input": query})

        print(result)
        
        return result["answer"]