"""
Multi-object detector for vehicles, signs, and billboards.
"""

import cv2
import logging
from ultralytics import YOLO
from typing import Dict, List, Any


class MultiObjectDetector:
    def __init__(self, model_path):
        logging.info("Loading YOLO model from: %s", model_path)
        self.model = YOLO(model_path)

        # Definicja kategorii obiektów (DODANO BILLBOARDY)
        self.object_categories = {
            "vehicles": {
                "classes": {2, 3, 5, 7},  # car, motorcycle, bus, truck
                "color": (255, 0, 0),  # Czerwony
                "name": "POJAZD",
            },
            "traffic_signs": {
                "classes": {11},  # stop sign
                "color": (0, 255, 0),  # Zielony
                "name": "ZNAK",
            },
            "billboards": {
                "classes": set(),  # Wykrywane geometrycznie, nie przez YOLO
                "color": (255, 165, 0),  # Pomarańczowy
                "name": "BILLBOARD",
            },
        }

        # Wszystkie dozwolone klasy YOLO (bez billboardów - te wykrywamy inaczej)
        self.allowed_classes = set()
        for category, config in self.object_categories.items():
            if category != "billboards":  # Billboardy nie są klasami YOLO
                self.allowed_classes.update(config["classes"])

        logging.info(
            "Object categories initialized (vehicles, signs, billboards)")
        logging.info("Allowed YOLO classes: %s", self.allowed_classes)

    def is_billboard_shape(self, bbox, frame_shape):
        """Sprawdza czy bounding box może być billboardem na podstawie kształtu"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height

        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_height * frame_width

        # Względny rozmiar
        rel_area = area / frame_area

        # Proporcje (billboardy mogą być poziome LUB pionowe)
        aspect_ratio = width / height if height > 0 else 0

        # Pozycja (billboardy zazwyczaj nie są na dole kadru)
        y_center = (y1 + y2) / 2
        relative_y = y_center / frame_height

        # Kryteria kształtu billboardu:
        # 1. Duży względny rozmiar
        size_ok = 0.01 < rel_area < 0.4  # 1-40% kadru

        # 2. Prostokątny kształt (może być poziomy LUB pionowy)
        # Poziomy: szerokość > wysokość (1.5-8x)
        # Pionowy: wysokość > szerokość (1.5-8x)
        horizontal_ok = 1.5 < aspect_ratio < 8.0  # Szeroki billboard
        # Wysoki billboard (1/8 - 2/3)
        vertical_ok = 0.125 < aspect_ratio < 0.67
        shape_ok = horizontal_ok or vertical_ok

        # 3. Nie na samym dole kadru
        position_ok = relative_y < 0.85  # Nie w dolnych 15% kadru

        # 4. Minimalny rozmiar bezwzględny
        min_size_ok = width > 50 and height > 50  # Zmniejszone minimum dla pionowych

        # Zwraca True tylko jeśli ma odpowiedni kształt (tekst sprawdzamy później)
        has_billboard_shape = size_ok and shape_ok and position_ok and min_size_ok

        if has_billboard_shape:
            billboard_type = "poziomy" if horizontal_ok else "pionowy"
            logging.debug(
                "Billboard shape detected (%s): area=%.3f, aspect=%.2f, y_pos=%.2f",
                billboard_type, rel_area, aspect_ratio, relative_y
            )

        return has_billboard_shape

    def detect_billboards_geometrically(
        self, frame, existing_detections, text_extractor, text_classifier
    ):
        """Wykrywa billboardy na podstawie kształtu I OBECNOŚCI TEKSTU"""
        billboard_detections = []

        # Sprawdź wszystkie wykrycia YOLO (nawet te spoza allowed_classes)
        results = self.model(frame)

        for result in results:
            for i, box in enumerate(result.boxes.xyxy):
                class_id = int(result.boxes.cls[i])
                class_name = self.model.names[class_id]
                conf = float(result.boxes.conf[i])

                # Pomiń jeśli to już wykryty pojazd lub znak
                bbox = tuple(map(int, box))
                is_already_detected = any(
                    abs(bbox[0] - det["bbox"][0]) < 20
                    and abs(bbox[1] - det["bbox"][1]) < 20
                    for det in existing_detections
                )

                if is_already_detected:
                    continue

                # Sprawdź czy ma kształt billboardu
                if self.is_billboard_shape(bbox, frame.shape) and conf > 0.2:

                    # SPRAWDŹ CZY MA TEKST I SKLASYFIKUJ GO
                    x1, y1, x2, y2 = bbox
                    roi = frame[y1:y2, x1:x2]
                    text = text_extractor.extract_text(roi)
                    # Tylko obiekty Z TEKSTEM są billboardami
                    if text and len(text.strip()) > 2:  # Minimum 3 znaki tekstu
                        try:
                            # KLASYFIKUJ TEKST
                            text_label = text_classifier.predict(text)
                            logging.debug(
                                "Billboard confirmed: shape OK + text found: '%s' (%s)",
                                text, text_label
                            )

                            billboard_detection = {
                                "bbox": bbox,
                                "class_id": class_id,
                                "class_name": class_name,
                                "confidence": conf,
                                "category": "billboards",
                                "config": self.object_categories["billboards"],
                                "area": ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                                / (frame.shape[0] * frame.shape[1]),
                                "detected_text": text,
                                "text_classification": text_label,  # DODAJ KLASYFIKACJĘ
                            }
                            billboard_detections.append(billboard_detection)
                        except Exception as e:
                            logging.debug("Text classification failed: %s", e)
                    else:
                        logging.debug(
                            "Shape looks like billboard but no text found - skipping"
                        )

        return billboard_detections

    def categorize_object(self, class_id):
        """Kategoryzuje obiekt na podstawie class_id"""
        if class_id not in self.allowed_classes:
            return None, None

        for category, config in self.object_categories.items():
            if category != "billboards" and class_id in config["classes"]:
                return category, config
        return None, None

    def detect(self, frame, text_extractor, text_classifier) -> List[Dict[str, Any]]:
        logging.debug("Detecting objects in frame")
        results = self.model(frame)
        detections = []
        height, width = frame.shape[:2]
        frame_area = width * height
        # Wykryj standardowe obiekty (pojazdy, znaki)
        for result in results:
            for i, box in enumerate(result.boxes.xyxy):
                class_id = int(result.boxes.cls[i])
                class_name = self.model.names[class_id]
                conf = float(result.boxes.conf[i])

                if conf < 0.3:
                    continue

                # Kategoryzacja obiektu (zwraca None dla nieznanych)
                category_result = self.categorize_object(class_id)
                if category_result[0] is None:
                    continue  # Pomiń nieznane obiekty

                category, config = category_result

                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                area = w * h
                rel_area = area / frame_area

                # Podstawowe filtry rozmiaru
                if not (0.0001 < rel_area < 0.8):
                    continue

                detection = {
                    "bbox": (x1, y1, x2, y2),
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "category": category,
                    "config": config,
                    "area": rel_area,
                }

                detections.append(detection)

        # Wykryj billboardy geometrycznie (PRZEKAŻ text_classifier)
        billboard_detections = self.detect_billboards_geometrically(
            frame, detections, text_extractor, text_classifier
        )
        detections.extend(billboard_detections)

        logging.debug(
            "Detected objects: %d (including %d billboards)",
            len(detections),
            len(billboard_detections),
        )
        return detections
