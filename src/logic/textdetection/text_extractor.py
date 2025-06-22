"""
Text extraction using OCR.
"""

import pytesseract


class TextExtractor:
    def __init__(self):
        self.langs = ["eng", "pol", "deu", "fra"]

    def extract_text(self, roi):
        """Wyciąga tekst z ROI używając OCR"""
        texts = []
        for lang in self.langs:
            try:
                text = pytesseract.image_to_string(roi, lang=lang).strip()
                if text and len(text) > 1:
                    texts.append(text)
            except Exception:
                continue

        return max(texts, key=len) if texts else ""
