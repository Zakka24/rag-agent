# src/smart_pdf_loader.py

# from typing import List, Optional
# from pathlib import Path

# import fitz  # PyMuPDF
# from pdf2image import convert_from_path
# from PIL import Image
# import pytesseract

# from langchain_core.documents import Document


# class SmartPDFLoader:

#     def __init__(self, file_path: str, lang: str = "ita"):
#         self.file_path = str(file_path)
#         self.lang = lang

#     def load(self) -> List[Document]:
#         pdf_path = Path(self.file_path)
#         if not pdf_path.exists():
#             raise FileNotFoundError(f"File not found: {self.file_path}")

#         pdf = fitz.open(self.file_path)
#         num_pages = len(pdf)

#         result_docs: List[Document] = []

#         for page_index in range(num_pages):
#             page_num = page_index + 1
#             page = pdf[page_index]

#             text = page.get_text().strip()

#             if text:
#                 result_docs.append(
#                     Document(
#                         page_content=text,
#                         metadata={
#                             "source": pdf_path.name,
#                             "page": page_num,
#                             "ocr": False,
#                         },
#                     )
#                 )
#             else:
#                 ocr_doc = self._ocr_page(pdf_path, page_num)
#                 if ocr_doc:
#                     result_docs.append(ocr_doc)

#         pdf.close()
#         return result_docs

#     def _ocr_page(self, pdf_path: Path, page_num: int) -> Optional[Document]:
#         images = convert_from_path(
#             str(pdf_path),
#             dpi=300,
#             first_page=page_num,
#             last_page=page_num,
#         )
#         if not images:
#             return None

#         image = images[0]

#         text = pytesseract.image_to_string(image, lang=self.lang)
#         text = text.strip()
#         if not text:
#             return None

#         return Document(
#             page_content=text,
#             metadata={
#                 "source": pdf_path.name,
#                 "page": page_num,
#                 "ocr": True,
#             },
#         )

from typing import List, Optional, Dict, Any
from pathlib import Path
import statistics

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output

from langchain_core.documents import Document


class SmartPDFLoader:
    def __init__(
        self,
        file_path: str,
        lang: str = "ita",
        dpi: int = 300,
        word_conf_threshold: float = 50,
        include_low_conf_tokens: bool = True,
        max_low_conf_tokens: int = 1000,
        tesseract_config: str = "",
    ):
        self.file_path = str(file_path)
        self.lang = lang
        self.dpi = dpi
        self.word_conf_threshold = word_conf_threshold
        self.include_low_conf_tokens = include_low_conf_tokens
        self.max_low_conf_tokens = max_low_conf_tokens
        self.tesseract_config = tesseract_config

    def load(self) -> List[Document]:
        pdf_path = Path(self.file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        pdf = fitz.open(self.file_path)
        num_pages = len(pdf)

        result_docs: List[Document] = []

        for page_index in range(num_pages):
            page_num = page_index + 1
            page = pdf[page_index]

            text = page.get_text().strip()

            if text:
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
                ocr_doc = self._ocr_page(pdf_path, page_num)
                if ocr_doc:
                    result_docs.append(ocr_doc)

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

        image = images[0]

        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=self.tesseract_config or "",
            output_type=Output.DICT,
        )

        words = data.get("text", [])
        confs = data.get("conf", [])
        lefts = data.get("left", [])
        tops = data.get("top", [])
        widths = data.get("width", [])
        heights = data.get("height", [])

        extracted_words: List[str] = []
        valid_confs: List[float] = []
        low_conf_tokens: List[Dict[str, Any]] = []

        for i in range(min(len(words), len(confs))):
            w = (words[i] or "").strip()
            if not w:
                continue

            try:
                c = float(confs[i])
            except Exception:
                continue

            # tesseract usa -1 per elementi non validi
            if c < 0:
                continue

            extracted_words.append(w)
            valid_confs.append(c)

            if c < self.word_conf_threshold and self.include_low_conf_tokens:
                low_conf_tokens.append(
                    {
                        "text": w,
                        "conf": c,  # 0..100
                        "bbox": [lefts[i], tops[i], widths[i], heights[i]],
                    }
                )

        text = " ".join(extracted_words).strip()
        if not text:
            return None

        # Metriche generali pagina
        mean_conf = statistics.mean(valid_confs) if valid_confs else None
        low_conf_ratio = (
            (sum(1 for c in valid_confs if c < self.word_conf_threshold) / len(valid_confs))
            if valid_confs
            else None
        )

        metadata: Dict[str, Any] = {
            "source": pdf_path.name,
            "page": page_num,
            "ocr": True,
            "dpi": self.dpi,
            "word_conf_threshold": self.word_conf_threshold,
            # confidenza aggregata pagina
            "ocr_conf_mean": (round(mean_conf, 2) if mean_conf is not None else None),  # 0..100
            "ocr_low_conf_ratio": (round(low_conf_ratio, 3) if low_conf_ratio is not None else None),
            # flag rapido
            "has_low_conf_words": bool(low_conf_tokens),
            "low_conf_word_count": len(low_conf_tokens),
        }

        # if self.include_low_conf_tokens:
        #     metadata["low_conf_words"] = low_conf_tokens[: self.max_low_conf_tokens]

        return Document(page_content=text, metadata=metadata)


# # src/smart_pdf_loader.py

# from typing import List, Optional
# from pathlib import Path
# import os
# import sys

# import fitz  # PyMuPDF
# from pdf2image import convert_from_path
# from PIL import Image

# import torch
# from transformers import AutoModelForCausalLM, AutoProcessor
# from langchain_core.documents import Document

# # DotsOCR (Qwen-VL utils)
# from qwen_vl_utils import process_vision_info


# class SmartPDFLoader:
#     """
#     Loader che:
#     - usa PyMuPDF per leggere il testo digitale dalle pagine
#     - se una pagina non ha testo (tipico scan), usa DotsOCR (Hugging Face) per fare OCR.
#     """

#     def __init__(
#         self,
#         file_path: str,
#         # model_path: str = os.getenv("DOTSOCR_PATH", "/weights/DotsOCR"),
#         model_path: str = "weights/DotsOCR",
#         lang: str = "ita",
#         dpi: int = 300,
#         max_new_tokens: int = 4096,
#     ):
#         """
#         Args:
#             file_path: percorso del PDF
#             model_path: path locale del modello DotsOCR (es. ./weights/DotsOCR)
#             dpi: DPI per conversione pagina->immagine
#             max_new_tokens: limite generazione per pagina
#         """
#         self.file_path = str(file_path)
#         self.model_path = str(model_path)
#         self.lang = lang
#         self.dpi = dpi
#         self.max_new_tokens = max_new_tokens

#         self._init_ocr_model()

#     def _init_ocr_model(self):
#         model_dir = Path(self.model_path)
#         if not model_dir.exists():
#             raise FileNotFoundError(f"Model path not found: {self.model_path}")

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_path,
#             dtype="auto",
#             device_map="auto",
#             trust_remote_code=True,
#             attn_implementation="sdpa"
#         )

#         self.processor = AutoProcessor.from_pretrained(
#             self.model_path,
#             trust_remote_code=True
#         )
#         self.model.eval()

#     def load(self) -> List[Document]:
#         pdf_path = Path(self.file_path)
#         if not pdf_path.exists():
#             raise FileNotFoundError(f"File not found: {self.file_path}")

#         pdf = fitz.open(self.file_path)
#         num_pages = len(pdf)

#         result_docs: List[Document] = []

#         for page_index in range(num_pages):
#             page_num = page_index + 1
#             page = pdf[page_index]

#             # 1) testo digitale
#             text = page.get_text().strip()

#             if text:
#                 result_docs.append(
#                     Document(
#                         page_content=text,
#                         metadata={"source": pdf_path.name, "page": page_num, "ocr": False},
#                     )
#                 )
#             else:
#                 # 2) OCR con DotsOCR
#                 print(f"   [DotsOCR] Elaborazione pagina {page_num}...")
#                 ocr_doc = self._ocr_page(pdf_path, page_num)
#                 if ocr_doc:
#                     result_docs.append(ocr_doc)

#         pdf.close()
#         return result_docs

#     def _ocr_page(self, pdf_path: Path, page_num: int) -> Optional[Document]:
#         images = convert_from_path(
#             str(pdf_path),
#             dpi=self.dpi,
#             first_page=page_num,
#             last_page=page_num,
#         )
#         if not images:
#             return None

#         image: Image.Image = images[0].convert("RGB")

#         prompt_text = (
#             "Esegui OCR su questa immagine e restituisci SOLO il testo estratto, "
#             "nell'ordine di lettura. Non aggiungere commenti o riassunti."
#         )

#         # Formato messaggi stile Qwen-VL (usato da DotsOCR)
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image},
#                     {"type": "text", "text": prompt_text},
#                 ],
#             }
#         ]

#         try:
#             # Applica chat template + prepara input multimodale
#             text = self.processor.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True,
#             )
#             image_inputs, video_inputs = process_vision_info(messages)

#             inputs = self.processor(
#                 text=[text],
#                 images=image_inputs,
#                 videos=video_inputs,
#                 padding=True,
#                 return_tensors="pt",
#             )

#             inputs = inputs.to("cuda")

#             with torch.inference_mode():
#                 generated_ids = self.model.generate(
#                     **inputs,
#                     max_new_tokens=self.max_new_tokens,
#                 )

#             generated_ids_trimmed = [
#                 out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#             ]
#             output_text = self.processor.batch_decode(
#                 generated_ids_trimmed,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False,
#             )

#             extracted = (output_text[0] if output_text else "").strip()
#             if not extracted:
#                 return None

#             return Document(
#                 page_content=extracted,
#                 metadata={
#                     "source": pdf_path.name,
#                     "page": page_num,
#                     "ocr": True,
#                     "ocr_model": str(Path(self.model_path).name),
#                 },
#             )
#         except Exception as e:
#             print(f"Errore durante OCR DotsOCR sulla pagina {page_num}: {e}")
#             return None
        


# if __name__ == '__main__':

#     dotsocr_path = os.getenv("DOTSOCR_PATH", "../weights/DotsOCR")

#     loader = SmartPDFLoader(
#         file_path='20220824 Felici ON11 2F Scansionato (3).pdf',
#         model_path=dotsocr_path,
#         lang="ita",
#         dpi=300,
#         max_new_tokens=4096,
#     )

#     loaded_documents = loader.load()

#     if not loaded_documents:
#         raise RuntimeError("OCR fallito: nessun testo estratto dal PDF.")