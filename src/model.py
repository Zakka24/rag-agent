from langchain_ollama import ChatOllama, OllamaEmbeddings

class Model:
    """
    Class to initialize and manage machine learning models for embeddings and chat.
    Ollama version (runs locally).
    """

    def __init__(
        self,
        embeddings_model: str,
        chat_model: str,
        base_url: str = "http://localhost:11434",
    ):
        """
        Args:
            embeddings_model (str): Nome del modello per gli embeddings (es. "nomic-embed-text").
            chat_model (str): Nome del modello LLM (es. "llama3" o "mistral").
            base_url (str): URL dell'istanza Ollama locale (default: http://localhost:11434).
            temperature (float): Temperatura del modello (default: 0.0).
        """
        self.embeddings_name = embeddings_model
        self.model_name = chat_model
        self.base_url = base_url

        self._instantiate_models()

    def _instantiate_models(self):
        self.embeddings_model = OllamaEmbeddings(
            model=self.embeddings_name,
            base_url=self.base_url
        )

        self.chat_model = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            num_ctx=16_384
        )
