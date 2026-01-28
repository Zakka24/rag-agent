from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings

class Model:
    """
    Class to initialize and manage machine learning models for embeddings and chat.
    Ollama version (runs locally).
    """

    def __init__(
        self,
        embeddings_model: str,
        chat_model: str,
        api_key: str,
        base_url: str = "http://localhost:11434",
    ):
        """
        Args:
            embeddings_model (str): Nome del modello per gli embeddings (es. "nomic-embed-text").
            chat_model (str): Nome del modello LLM (es. "llama3" o "mistral").
            base_url (str): URL dell'istanza Ollama locale.
        """
        self.embeddings_name = embeddings_model
        self.model_name = chat_model
        self.api_key = api_key
        self.base_url = base_url

        self._instantiate_models()

    def _instantiate_models(self):
        self.embeddings_model = OllamaEmbeddings(
            model=self.embeddings_name,
            base_url=self.base_url
        )

        self.chat_model = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            max_retries= 0
        )
