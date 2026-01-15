from typing import List, Optional, Dict, Any
from pathlib import Path
import statistics

import fitz 
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