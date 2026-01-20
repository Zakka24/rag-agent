import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import deepdoctection as dd
from langchain_core.documents import Document
import pytesseract
from PIL import Image

class SmartPDFLoader:
    def __init__(
        self,
        file_path: str,
        dpi: int = 300,
        lang: str = "ita"
    ):
        self.file_path = str(file_path)
        self.dpi = dpi
        self.lang = lang

        print("Inizializzazione DeepDoctection (Analyzer)...")
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
                print(f"  -> Pagina {page_num}: Testo nativo assente. Avvio IBRIDO (Tesseract + DeepDoctection)...")
                hybrid_doc = self._process_page_hybrid(pdf_path, page_num)
                if hybrid_doc:
                    result_docs.append(hybrid_doc)
                else:
                    print(f"  -> Pagina {page_num}: Nessun testo trovato.")

        pdf.close()
        return result_docs

    def _process_page_hybrid(self, pdf_path: Path, page_num: int) -> Optional[Document]:
        temp_pdf_path = None
        try:
            images = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                first_page=page_num,
                last_page=page_num,
            )
            
            if not images:
                return None

            pil_image = images[0]

            raw_text = pytesseract.image_to_string(pil_image, lang=self.lang)

            tables_content = ""
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                pil_image.save(temp_pdf.name, "PDF", resolution=self.dpi)
                temp_pdf_path = temp_pdf.name

            try:
                df = self.analyzer.analyze(path=temp_pdf_path)
                df.reset_state()
                
                doc_result = next(iter(df))
                
                if doc_result.tables:
                    tables_content += "\n\n"
                    tables_content += "--- TABELLE STRUTTURATE ---\n"
                    tables_content += "Usa questi dati se il testo sopra è disallineato.\n"
                    
                    for i, table in enumerate(doc_result.tables):
                        tables_content += f"\n[Tabella {i+1}]\n"
                        
                        if hasattr(table, "csv") and table.csv:
                            try:
                                for row in table.csv:
                                    # Pulizia celle vuote
                                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                                    tables_content += " | ".join(clean_row) + "\n"
                            except Exception as e:
                                tables_content += f"(Errore parsing righe CSV: {e})\n"
                        else:
                            tables_content += "(Struttura tabella rilevata ma non convertibile in CSV)\n"
                        
                        tables_content += "-"*30 + "\n"

            except StopIteration:
                pass
            except Exception as e:
                print(f"Warn: Errore parziale DeepDoctection su pagina {page_num}: {e}")

            full_content = raw_text + tables_content

            if not full_content.strip():
                return None

            metadata: Dict[str, Any] = {
                "source": pdf_path.name,
                "page": page_num,
                "ocr": True,
                "ocr_engine": "hybrid_tesseract_dd",
                "has_tables": len(tables_content) > 0
            }

            return Document(page_content=full_content, metadata=metadata)

        except Exception as e:
            print(f"Errore CRITICO pagina {page_num}: {e}")
            return None
        finally:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                except OSError:
                    pass

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