import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import deepdoctection as dd
from langchain_core.documents import Document
import torch

class SmartPDFLoader:
    def __init__(
        self,
        file_path: str,
        dpi: int = 300,
    ):
        self.file_path = str(file_path)
        self.dpi = dpi

        # if torch.cuda.is_available():
        #     print(f"GPU Rilevata: {torch.cuda.get_device_name(0)}")
        # else:
        #     print("Nessuna GPU rilevata. L'elaborazione sarà lenta (CPU).")
        
        print("Inizializzazione DeepDoctection (potrebbe richiedere tempo la prima volta)...")
        self.analyzer = dd.get_dd_analyzer()

    def load(self) -> List[Document]:
        pdf_path = Path(self.file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        pdf = fitz.open(self.file_path)
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
            else:
                print(f"  -> Pagina {page_num}: Testo nativo assente/insufficiente. Avvio OCR DeepDoctection...")
                ocr_doc = self._ocr_page(pdf_path, page_num)
                if ocr_doc:
                    result_docs.append(ocr_doc)
                else:
                    print(f"  -> Pagina {page_num}: Nessun testo trovato nemmeno con OCR.")

        pdf.close()
        return result_docs

    def _ocr_page(self, pdf_path: Path, page_num: int) -> Optional[Document]:
        images = convert_from_path(
            str(pdf_path),
            dpi=self.dpi,
            first_page=page_num,
            last_page=page_num,
        )
        
        if not images:
            return None

        pil_image = images[0]
        temp_pdf_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                pil_image.save(temp_pdf.name, "PDF", resolution=self.dpi)
                temp_pdf_path = temp_pdf.name

            # Analisi
            df = self.analyzer.analyze(path=temp_pdf_path)
            df.reset_state()
            
            try:
                doc_result = next(iter(df))
            except StopIteration:
                return None
            
            extracted_text = doc_result.text
            
            if doc_result.tables:
                extracted_text += "\n\n--- TABELLE ESTRATTE (FORMATO CSV) ---\n"
                
                for i, table in enumerate(doc_result.tables):
                    extracted_text += f"\n[Tabella {i+1}]\n"
                    
                    if hasattr(table, "csv") and table.csv:
                        try:
                            for row in table.csv:
                                clean_row = [str(cell).strip() if cell else "" for cell in row]
                                extracted_text += " | ".join(clean_row) + "\n"
                        except Exception as e:
                            print(f"Warn: Errore estrazione CSV tabella: {e}")
                    else:
                        extracted_text += "(Tabella rilevata ma struttura complessa non risolta)\n"
                    
                    extracted_text += "\n" + "-"*30 + "\n"

            if not extracted_text or not extracted_text.strip():
                return None
            
            metadata: Dict[str, Any] = {
                "source": pdf_path.name,
                "page": page_num,
                "ocr": True,
                "ocr_engine": "deepdoctection",
                # Aggiungiamo un flag per sapere se ci sono tabelle
                "has_tables": len(doc_result.tables) > 0 
            }

            return Document(page_content=extracted_text, metadata=metadata)

        except Exception as e:
            print(f"Errore critico DeepDoctection pagina {page_num}: {e}")
            return None
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                except OSError:
                    pass

def main():
    pdf_filename = "Serra Dario_superficie_servitú_rep.65.631_racc.24.117.pdf"

    if not os.path.exists(pdf_filename):
        print(f"ERRORE: File {pdf_filename} non trovato.")
        return

    loader = SmartPDFLoader(file_path=pdf_filename)
    docs = loader.load()

    print("\n" + "="*60)
    print(f"ELABORAZIONE COMPLETATA. Pagine estratte: {len(docs)}")
    print("="*60 + "\n")
    
    for doc in docs:
        status = "OCR (DeepDoctection)" if doc.metadata['ocr'] else "Nativo (Veloce)"
        print(f"--- Pagina {doc.metadata['page']} [{status}] ---")
        print(doc.page_content + "\n")

if __name__ == '__main__':
    main()