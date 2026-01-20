from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path
from uuid import uuid4
import os

from src.model import Model
from src.smart_pdf_loader import SmartPDFLoader


class Ingestor:
    """
        Class to handle the ingestion of pdf documents into vector store.
    """

    def __init__(self, file_name: str, model: Model, chunk_size=1000, chunk_overlap: int = 150, user_id: str | None=None):
        """
        Initialize the Ingestor class and immediately instantiate vector store.

        Args:
            file_name (str): The name of the file to ingest.
            model (Model): The model used for embeddings.
            chunk_size (int): The size of each chunk for splitting documents. Default is 1000.
            chunk_overlap (int): The number of characters to overlap between chunks. Default is 300.
            user_id: create dedicated folders
        """
        self.model = model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.user_id = user_id

        base_folder = Path(__file__).parent.parent
        base_data_folder = base_folder / 'data'
        base_db_folder = base_folder / 'db'

        file_stem = Path(file_name).stem

        # Cartelle separate per utente
        if user_id is not None:
            self.data_folder = base_data_folder / user_id
            self.persist_directory = base_db_folder / user_id / file_stem / 'chroma_langchain_db'
        else:
            self.data_folder = base_data_folder
            self.persist_directory = base_db_folder / file_stem / 'chroma_langchain_db'

        self.file_path = self.data_folder / file_name

        # Creazione cartelle se non esistono
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.persist_directory.parent.mkdir(parents=True, exist_ok=True)

        self._instantiate_vector_store()

    def _instantiate_vector_store(self):
        """
        Instantiate and initialize the vector store using Chroma with the model's embedding.
        This method is called during the initialization of the Ingestor class.
        """
        # if self.persist_directory.exists():
        #     shutil.rmtree(self.persist_directory)

        self.vector_store = Chroma(collection_name="documents",
                                   embedding_function=self.model.embeddings_model,
                                   persist_directory=str(self.persist_directory.absolute()))

    def ingest_file(self):
        """
        Ingest PDF file into the vector store by performing the following steps:
        - Check that the file is a PDF.
        - Load the PDF file.
        - Split the content of the PDF into chunks.
        - Add the chunks to the vector store.

        Raises:
            ValueError: If the file is not a PDF.
        """
        if self.file_path.suffix != '.pdf':
            raise ValueError('The file must be a pdf.')

        print(f"Caricamento documento: {self.file_path.name}")
        loader = SmartPDFLoader(str(self.file_path))
        loaded_documents = loader.load()        

        if not loaded_documents:
            raise RuntimeError("OCR fallito: nessun testo estratto dal PDF.")
        
        separators = [
            "\n\n",
            "--- TABELLE",
            "\n",
            ". ",
            " ",
            ""
        ]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap, separators=separators, keep_separator=True)
        documents = text_splitter.split_documents(loaded_documents)
        documents = [d for d in documents if d.page_content and d.page_content.strip()]

        if not documents:
            raise RuntimeError("Nessun chunk valido generato (testo vuoto).")

        for doc in documents:
            page = doc.metadata.get("page")
            doc.metadata.setdefault("source", self.file_path.name)
            doc.metadata.setdefault("ocr", False)
            if not doc.page_content.strip().startswith(f"[PAGINA {page}]"):
                doc.page_content = f"[PAGINA {page}]\n{doc.page_content}"

        self.documents = loaded_documents

        # debug_file_name = f"{self.file_path.stem}_debug_chunks.txt"
        # debug_path = self.data_folder / debug_file_name
        
        # print(f"Generazione file di debug chunk: {debug_path}")
        # try:
        #     with open(debug_path, "w", encoding="utf-8") as f:
        #         f.write(f"REPORT DEBUG CHUNKS per: {self.file_path.name}\n")
        #         f.write(f"Totale chunk generati: {len(documents)}\n")
        #         f.write("="*60 + "\n\n")
                
        #         for i, doc in enumerate(documents):
        #             f.write(f"--- CHUNK {i+1} ---\n")
        #             f.write(f"METADATI: {doc.metadata}\n")
        #             f.write(f"LUNGHEZZA: {len(doc.page_content)} caratteri\n")
        #             f.write("-" * 20 + " INIZIO CONTENUTO " + "-" * 20 + "\n")
        #             f.write(doc.page_content)
        #             f.write("\n" + "-" * 20 + " FINE CONTENUTO " + "-" * 20 + "\n")
        #             f.write("\n\n")
                    
        #     print(f"--> File di debug salvato correttamente.")
        # except Exception as e:
        #     print(f"Attenzione: Impossibile salvare file di debug: {e}")

        uuids = [str(uuid4()) for _ in range(len(documents))]

        self.vector_store.add_documents(documents=documents, ids=uuids)