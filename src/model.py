import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

class Model:
    def __init__(
        self,
        embeddings_model: str,
        chat_model: str,
        base_url: str, 
        temperature: float = 0.0
    ):
        self.embeddings_name = embeddings_model
        self.model_name = chat_model
        self.base_url = base_url
        self.temperature = temperature

        self._instantiate_models()

    def _instantiate_models(self):
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=self.embeddings_name,

        )

        self.chat_model = ChatOpenAI(
            model=self.model_name,
            openai_api_base=self.base_url,
            openai_api_key="token-finto",
            temperature=self.temperature,
            max_tokens=16_384
        )