# import os
# import torch
# from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEmbeddings
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# class Model:
#     """
#     Class to initialize and manage machine learning models for embeddings and chat.
#     Local Execution Version.
#     """

#     def __init__(
#         self,
#         embeddings_model: str,
#         chat_model: str,
#         hf_token: str | None = None,
#         hf_task: str = "text-generation",
#     ):
#         """
#         Args:
#             embeddings_model: Nome o percorso del modello di embeddings.
#             chat_model: Percorso assoluto alla cartella del modello locale.
#             hf_token: Token HF (opzionale se il modello è già scaricato).
#             hf_task: di default "text-generation".
#         """
#         self.embeddings = embeddings_model
#         self.model_path = chat_model
#         self.hf_token = hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
#         self.hf_task = hf_task

#         self._instantiate_models()

#     def _instantiate_models(self):
#         print(f"Caricamento Embeddings: {self.embeddings}")
#         self.embeddings_model = HuggingFaceEmbeddings(
#             model_name=self.embeddings,
#         )

#         print(f"Caricamento Modello LLM locale da: {self.model_path}")
        
#         tokenizer = AutoTokenizer.from_pretrained(
#             self.model_path,
#             token=self.hf_token,
#             trust_remote_code=True
#         )

#         model = AutoModelForCausalLM.from_pretrained(
#             self.model_path,
#             device_map="auto", 
#             dtype="auto", 
#             token=self.hf_token,
#             trust_remote_code=True
#         )

#         pipe = pipeline(
#             task=self.hf_task,
#             temperature=0.1,
#             model=model,
#             tokenizer=tokenizer,
#             max_new_tokens=16_364,
#         )

#         llm = HuggingFacePipeline(pipeline=pipe)

#         self.chat_model = ChatHuggingFace(llm=llm)
        
#         print("Modello caricato correttamente.")


# import os
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # Chat via Hugging Face Inference API / Inference Endpoint
# from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
# from langchain_huggingface.chat_models import ChatHuggingFace
# from transformers import AutoModelForCausalLM, AutoTokenizer


# class Model:
#     """
#     Class to initialize and manage machine learning models for embeddings and chat.
#     Hugging Face version (instead of Ollama).
#     """

#     def __init__(
#         self,
#         embeddings_model: str,
#         chat_model: str,
#         hf_token: str | None = None,
#         hf_task: str = "text-generation",
#     ):
#         self.embeddings = embeddings_model
#         self.model = chat_model
#         self.hf_token = hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
#         self.hf_task = hf_task

#         self._instantiate_models()

#     def _instantiate_models(self):

#         self.embeddings_model = HuggingFaceEmbeddings(
#             model_name=self.embeddings,
#         )

#         llm = HuggingFaceEndpoint(
#             repo_id=self.model,
#             task=self.hf_task,
#             huggingfacehub_api_token=self.hf_token,
#             temperature=0.0,
#             max_new_tokens=16_392,
#         )

#         # Wrapping in chat model così lavora bene con ChatPromptTemplate
#         self.chat_model = ChatHuggingFace(llm=llm)



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
        temperature: float = 0.0
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
            num_ctx=65_536
        )
