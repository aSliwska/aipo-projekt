"""
License plate recognition using OpenALPR.
"""

import cv2
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List


class PlateRecognizer:
    def __init__(self, country="eu", alpr_path=None):
        logging.info("Initializing PlateRecognizer (CLI mode)")
        if alpr_path is None:
            alpr_path = "alpr"
        self.alpr_path = alpr_path
        self.country = country

    def recognize(self, image_roi) -> List[str]:
        """Rozpoznaje tablice bezpośrednio z ROI (numpy array)"""
        logging.debug("Recognizing plates in ROI")
        try:
            # Użyj tempfile do stworzenia tymczasowego pliku
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_path = tmp_file.name
                cv2.imwrite(temp_path, image_roi)

            # Uruchom ALPR
            result = subprocess.run(
                [self.alpr_path, "-c", self.country, "-j", temp_path],
                capture_output=True,
                text=True,
                check=True,
            )
            output = json.loads(result.stdout)
            plates = [plate["plate"] for plate in output.get("results", [])]

            # Usuń plik tymczasowy
            Path(temp_path).unlink()

            logging.debug("Recognized plates: %s", plates)
            return plates

        except Exception as e:
            logging.error("Error running alpr CLI: %s", e)
            # Usuń plik tymczasowy w przypadku błędu
            try:
                Path(temp_path).unlink()
            except Exception:
                pass
            return []

    def __del__(self):
        pass
