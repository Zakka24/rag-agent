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

    def __init__(self, model: Model, chunk_size=1000, chunk_overlap: int = 150, user_id: str | None=None):
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
        self.base_data_folder = base_folder / 'data' / 'local'
        self.persist_directory = base_folder / 'db' / 'local' / user_id / str(uuid4()) / 'chroma_db'

    def _instantiate_vector_store(self):
        """
        Instantiate and initialize the vector store using Chroma with the model's embedding.
        This method is called during the initialization of the Ingestor class.
        """
        self.persist_directory.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store = Chroma(collection_name="documents",
                                   embedding_function=self.model.embeddings_model,
                                   persist_directory=str(self.persist_directory.absolute()))

    def ingest_files(self, file_paths: list[Path]):
        """
        Ingest PDF file into the vector store by performing the following steps:
        - Check that the file is a PDF.
        - Load the PDF file.
        - Split the content of the PDF into chunks.
        - Add the chunks to the vector store.

        Raises:
            ValueError: If the file is not a PDF.
        """
        self._instantiate_vector_store()
        all_splits = []

        for file_path in file_paths:
            if file_path.suffix.lower() != '.pdf':
                print(f"Saltato file non PDF: {file_path.name}")
                continue

            print(f"[{self.user_id}] Elaborazione file: {file_path.name}")

            try:
                loader = SmartPDFLoader(str(file_path))
                raw_docs = loader.load()
            except Exception as e:
                print(f"Errore caricamento {file_path.name}: {e}")
                continue

            if not raw_docs:
                print(f"Attenzione: Nessun testo estratto da {file_path.name}")
                continue

            documents = []

            for doc in raw_docs:
                page = doc.metadata['page']
                doc.metadata.setdefault("source", file_path.name)

                documents.append(doc)

            text_splitter = RecursiveCharacterTextSplitter( 
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                strip_whitespace=True
            )
            chunks = text_splitter.split_documents(documents)
            for chunk in chunks:
                page = chunk.metadata.get("page", "N/A")
                header = f"[FILE: {file_path.name} | PAGINA {page}]"
                chunk.page_content = f"{header}\n{chunk.page_content}"

            all_splits.extend(chunks)

        if not all_splits:
            raise RuntimeError("Nessun chunk valido generato dai file forniti.")
        
        print(f"[{self.user_id}] Inserimento di {len(all_splits)} chunks nel Vector DB...")

        self.documents = all_splits

        batch_size = 500
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size]
            uuids = [str(uuid4()) for _ in range(len(batch))]
            self.vector_store.add_documents(documents=batch, ids=uuids)
            
        print(f"[{self.user_id}] Ingestione completata.")