import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import deepdoctection as dd
from langchain_core.documents import Document
from PIL import Image
import numpy as np
import easyocr

class SmartPDFLoader:
    def __init__(
        self,
        file_path: str,
        dpi: int = 300,
        lang: str = "it",
        batch_size = 10
    ):
        self.file_path = str(file_path)
        self.dpi = dpi
        self.lang = lang
        self.batch_size = batch_size

        print("Inizializzazione DeepDoctection")
        self.analyzer = dd.get_dd_analyzer()

        print("Inizializzazione EasyOcr")
        self.reader = easyocr.Reader([self.lang], gpu=True)

    def load(self) -> List[Document]:
        pdf_path = Path(self.file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        is_digital = self._check_if_digital(self.file_path)

        if is_digital:
            return self._load_digital(self.file_path)
        else:
            return self._load_scanned_batched(self.file_path)
        

    def _load_digital(self, path: str) -> List[Document]:
        result_docs: List[Document] = []
        pdf_path = Path(path)

        pdf = fitz.open(path)
        num_pages = len(pdf)
        result_docs: List[Document] = []

        print(f"Analisi documento: {num_pages} pagine totali.")

        for page_index in range(num_pages):
            page_num = page_index + 1
            page = pdf[page_index]

            text = page.get_text().strip()

            if len(text) > 10:
                result_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                            "ocr": False,
                        },
                    )
                )

        pdf.close()
        return result_docs

    def _load_scanned_batched(self, path: str) -> List[Document]:
        """
        Percorso ottimizzato per scansioni:
        1. Converte N pagine in immagini IN PARALLELO (CPU Multi-core)
        2. Elabora le immagini con GPU in sequenza rapida
        """
        results = []
        
        with fitz.open(path) as doc:
            total_pages = len(doc)
        
        print(f"Elaborazione Scansione: {total_pages} pagine totali in batch da {self.batch_size}.")

        for i in range(0, total_pages, self.batch_size):
            start_page = i + 1
            end_page = min(i + self.batch_size, total_pages)
            
            print(f"  -> Preparazione batch {start_page}-{end_page}...")
            
            try:
                images = convert_from_path(
                    path,
                    dpi=self.dpi,
                    first_page=start_page,
                    last_page=end_page,
                    thread_count=4 
                )
            except Exception as e:
                print(f"Errore conversione batch {start_page}-{end_page}: {e}")
                continue

            for idx, pil_image in enumerate(images):
                page_num = start_page + idx
                print(f"     -> GPU Processing Pagina {page_num}...")
                
                doc = self._process_image_gpu(pil_image, Path(path).name, page_num)
                if doc:
                    results.extend(doc)
                
                del pil_image
            
            del images

        return results
    
    def _check_if_digital(self, path: str) -> bool:
        """Controlla se la prima pagina contiene testo nativo."""
        try:
            with fitz.open(path) as doc:
                if len(doc) > 0:
                    text = doc[0].get_text().strip()
                    return len(text) > 10
        except Exception:
            return False
        return False

    def _process_image_gpu(self, pil_image, filename: str, page_num: int) -> Optional[Document]:
        """Logica di estrazione singola immagine (EasyOCR + DeepDoctection)"""
        generated_docs = []

        try:
            img_array = np.array(pil_image)
            text_list = self.reader.readtext(img_array, detail=0, paragraph=True)
            raw_text = "\n".join(text_list)

            if raw_text.strip():
                generated_docs.append(Document(
                    page_content=raw_text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "ocr": True,
                        "type": "text"
                    }
                ))

            tables_content = ""
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                pil_image.save(temp_pdf.name, "PDF", resolution=self.dpi)
                temp_pdf_path = temp_pdf.name

            try:
                df = self.analyzer.analyze(path=temp_pdf_path)
                df.reset_state()
                doc_result = next(iter(df))
                
                if doc_result.tables:
                    for i, table in enumerate(doc_result.tables):
                        tables_content += f"\n[Tabella {i+1}]\n"

                        if hasattr(table, "csv") and table.csv:
                            for row in table.csv:
                                clean_row = [str(cell).strip() if cell else "" for cell in row]
                                tables_content += " | ".join(clean_row) + "\n"

                        generated_docs.append(Document(
                            page_content=tables_content,
                            metadata={
                                "source": filename,
                                "page": page_num,
                                "ocr": True,
                                "type": "table"
                            }
                        ))
            except StopIteration:
                pass
            except Exception as e:
                print(f"Warn: Errore tabelle pag {page_num}: {e}")
            finally:
                 if os.path.exists(temp_pdf_path):
                    try:
                        os.remove(temp_pdf_path)
                    except OSError: pass

            return generated_docs

        except Exception as e:
            print(f"Errore critico GPU pagina {page_num}: {e}")
            return None

# def main():
#     pdf_filename = "Serra Dario_superficie_servitú_rep.65.631_racc.24.117 (2).pdf"

#     if not os.path.exists(pdf_filename):
#         print(f"ERRORE: File {pdf_filename} non trovato.")
#         return

#     loader = SmartPDFLoader(file_path=pdf_filename)
#     docs = loader.load()

#     print("\n" + "="*60)
#     print(f"ELABORAZIONE COMPLETATA. Pagine estratte: {len(docs)}")
#     print("="*60 + "\n")
    
#     for doc in docs:
#         status = "OCR (DeepDoctection)" if doc.metadata['ocr'] else "Nativo (Veloce)"
#         print(f"--- Pagina {doc.metadata['page']} [{status}] ---")
#         print(doc.page_content + "\n")

# if __name__ == '__main__':
#     main()