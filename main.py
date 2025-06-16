import cv2
import json
import logging
import os
import pytesseract
import requests
import subprocess
import tempfile

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from ultralytics import YOLO
from urllib.parse import quote
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


os.environ["TESSDATA_PREFIX"] = "/usr/local/share/"

# Configure logging for verbose output
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ========== Klasyfikacja tekstu ==========
class TextClassifier:
    def __init__(self):
        logging.debug("Initializing TextClassifier")
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression()
        self.trained = False

    def train(self, texts, labels):
        logging.info("Training TextClassifier with texts: %s", texts)
        X_train = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X_train, labels)
        self.trained = True
        logging.info("TextClassifier training complete")

    def predict(self, text):
        logging.debug("Predicting label for text: %s", text)
        if not self.trained:
            raise ValueError("Model nie został wytrenowany.")
        vect = self.vectorizer.transform([text])
        prediction = self.classifier.predict(vect)[0]
        logging.debug("Prediction result: %s", prediction)
        return prediction


# ========== Multi-Object Detector (Z BILLBOARDAMI) ==========
class MultiObjectDetector:
    def __init__(self, model_path):
        logging.info("Loading YOLO model from: %s", model_path)
        self.model = YOLO(model_path)
        
        # Definicja kategorii obiektów (DODANO BILLBOARDY)
        self.object_categories = {
            'vehicles': {
                'classes': {2, 3, 5, 7},  # car, motorcycle, bus, truck
                'color': (255, 0, 0),  # Czerwony
                'name': 'POJAZD'
            },
            'traffic_signs': {
                'classes': {11},  # stop sign
                'color': (0, 255, 0),  # Zielony
                'name': 'ZNAK'
            },
            'billboards': {
                'classes': set(),  # Wykrywane geometrycznie, nie przez YOLO
                'color': (255, 165, 0),  # Pomarańczowy
                'name': 'BILLBOARD'
            }
        }
        
        # Wszystkie dozwolone klasy YOLO (bez billboardów - te wykrywamy inaczej)
        self.allowed_classes = set()
        for category, config in self.object_categories.items():
            if category != 'billboards': #Billboardy nie są klasami YOLO
                self.allowed_classes.update(config['classes'])
        
        logging.info("Object categories initialized (vehicles, signs, billboards)")
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
        vertical_ok = 0.125 < aspect_ratio < 0.67  # Wysoki billboard (1/8 - 2/3)
        shape_ok = horizontal_ok or vertical_ok
        
        # 3. Nie na samym dole kadru
        position_ok = relative_y < 0.85  # Nie w dolnych 15% kadru
        
        # 4. Minimalny rozmiar bezwzględny
        min_size_ok = width > 50 and height > 50  # Zmniejszone minimum dla pionowych
        
        # Zwraca True tylko jeśli ma odpowiedni kształt (tekst sprawdzamy później)
        has_billboard_shape = size_ok and shape_ok and position_ok and min_size_ok
        
        if has_billboard_shape:
            billboard_type = "poziomy" if horizontal_ok else "pionowy"
            logging.debug(f"Billboard shape detected ({billboard_type}): area={rel_area:.3f}, aspect={aspect_ratio:.2f}, y_pos={relative_y:.2f}")
        
        return has_billboard_shape

    def detect_billboards_geometrically(self, frame, existing_detections, text_extractor, text_classifier):
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
                    abs(bbox[0] - det['bbox'][0]) < 20 and abs(bbox[1] - det['bbox'][1]) < 20
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
                            logging.debug(f"Billboard confirmed: shape OK + text found: '{text}' ({text_label})")
                            
                            billboard_detection = {
                                'bbox': bbox,
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': conf,
                                'category': 'billboards',
                                'config': self.object_categories['billboards'],
                                'area': ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (frame.shape[0] * frame.shape[1]),
                                'detected_text': text,
                                'text_classification': text_label  # DODAJ KLASYFIKACJĘ
                            }
                            billboard_detections.append(billboard_detection)
                        except Exception as e:
                            logging.debug(f"Text classification failed: {e}")
                    else:
                        logging.debug(f"Shape looks like billboard but no text found - skipping")
        
        return billboard_detections

    def categorize_object(self, class_id):
        """Kategoryzuje obiekt na podstawie class_id"""
        if class_id not in self.allowed_classes:
            return None, None
            
        for category, config in self.object_categories.items():
            if category != 'billboards' and class_id in config['classes']:
                return category, config
        return None, None

    def detect(self, frame, text_extractor, text_classifier):
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
                    'bbox': (x1, y1, x2, y2),
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'category': category,
                    'config': config,
                    'area': rel_area
                }
                
                detections.append(detection)
        
        # Wykryj billboardy geometrycznie (PRZEKAŻ text_classifier)
        billboard_detections = self.detect_billboards_geometrically(frame, detections, text_extractor, text_classifier)
        detections.extend(billboard_detections)
                
        logging.debug("Detected objects: %d (including %d billboards)", len(detections), len(billboard_detections))
        return detections


# ========== Plate Recognizer =========
class PlateRecognizer:
    def __init__(self, country="eu", alpr_path=None):
        logging.info("Initializing PlateRecognizer (CLI mode)")
        if alpr_path is None:
            alpr_path = "/mnt/c/Users/wewek/Desktop/aipo/openalpr/src/build/alpr"
        self.alpr_path = alpr_path
        self.country = country

    def recognize(self, image_roi):
        """Rozpoznaje tablice bezpośrednio z ROI (numpy array)"""
        logging.debug("Recognizing plates in ROI")
        try:
            # Użyj tempfile do stworzenia tymczasowego pliku
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
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
            except:
                pass
            return []

    def __del__(self):
        pass


# ========== Text Extractor ==========
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
            except:
                continue

        return max(texts, key=len) if texts else ""


# ========== Geolocation ==========
def geolocate_text(text):
    logging.info("Geolocating text: %s", text)
    query = quote(text)
    url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
    headers = {"User-Agent": "GeoguessrApp/1.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                lat, lon = data[0]["lat"], data[0]["lon"]
                logging.info("Geolocation result: %s, %s", lat, lon)
                return lat, lon
    except Exception as e:
        logging.error("Error during geolocation: %s", e)
    return None, None


# ========== Main Function ==========
def main():
    logging.info("Starting video processing (vehicles, signs, billboards)")
    video_path = input("Podaj ścieżkę do pliku wideo: ").strip()
    video_file = Path(video_path)
    
    if not video_file.is_file():
        logging.error("Nie znaleziono pliku wideo: %s", video_path)
        logging.info("Nie znaleziono pliku wideo.")
        return

    # Inicjalizacja komponentów
    text_clf = TextClassifier()
    train_texts = ["Warszawa", "60 km/h", "Euro", "Berlin", "Stop", "DUR", "USD", "McDonald's", "Shell", "Coca-Cola"]
    train_labels = ["city", "sign", "currency", "city", "sign", "sign", "currency", "brand", "brand", "brand"]
    text_clf.train(train_texts, train_labels)

    detector = MultiObjectDetector("yolov8n.pt")
    plate_recognizer = PlateRecognizer()
    text_extractor = TextExtractor()
    
    # Sprawdź dozwolone klasy modelu
    logging.info("Wykrywane klasy w modelu:")
    for class_id, class_name in detector.model.names.items():
        if class_id in detector.allowed_classes:
            logging.info(f"{class_id}: {class_name}")
    logging.info("+ Billboardy (prostokąty z tekstem)")
    
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        logging.error("Nie można otworzyć pliku wideo: %s", video_path)
        return

    # POLICZ CAŁKOWITĄ LICZBĘ RAMEK
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    
    frame_count = 0
    processed_frames = 0
    
    # Utwórz katalogi dla różnych kategorii
    output_base = Path("outputs")
    output_dirs = {
        'vehicles': output_base / "cars",
        'traffic_signs': output_base / "signs",
        'billboards': output_base / "billboards",
        'mixed': output_base / "mixed"
    }
    
    # Utwórz wszystkie katalogi
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    plates_txt_path = output_base / "plates.txt"
    seen_plates = set()

    with tqdm(total=total_frames, desc="Processing video", unit="frames", position=0, leave=True) as pbar, logging_redirect_tqdm():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    tqdm.write("No more frames to read. Exiting loop.")
                    break

                frame_count += 1
                pbar.update(1)
                
                if frame_count % 10 != 0:
                    continue

                processed_frames += 1
                pbar.set_description(f"Processing frame {frame_count}/{total_frames} (analyzed: {processed_frames})")
                
                detections = detector.detect(frame, text_extractor, text_clf)
                frame_info = []
                
                # Kategoryzuj wykrycia w ramce
                vehicle_detections = []
                sign_detections = []
                billboard_detections = []
                
                for detection in detections:
                    if detection['category'] == 'vehicles':
                        vehicle_detections.append(detection)
                    elif detection['category'] == 'traffic_signs':
                        sign_detections.append(detection)
                    elif detection['category'] == 'billboards':
                        billboard_detections.append(detection)

                # SPRAWDŹ TABLICE REJESTRACYJNE DLA POJAZDÓW
                vehicles_with_plates = []
                for detection in vehicle_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    roi = frame[y1:y2, x1:x2]
                    plates = plate_recognizer.recognize(roi)
                    
                    if plates:
                        detection['plates'] = plates
                        vehicles_with_plates.append(detection)
                
                vehicle_detections = vehicles_with_plates

                # Przetwórz wszystkie wykrycia
                all_detections = vehicle_detections + sign_detections + billboard_detections
                
                for detection in all_detections:
                    x1, y1, x2, y2 = detection['bbox']
                    category = detection['category']
                    config = detection['config']
                    class_name = detection['class_name']
                    conf = detection['confidence']
                    
                    # Wyciągnij ROI
                    roi = frame[y1:y2, x1:x2]
                    
                    # Rysuj prostokąt z odpowiednim kolorem
                    cv2.rectangle(frame, (x1, y1), (x2, y2), config['color'], 2)
                    
                    label_text = f"{config['name']}: {class_name}"
                    additional_info = []
                    
                    # Analiza specyficzna dla kategorii
                    if category == 'vehicles':
                        plates = detection.get('plates', [])
                        if plates:
                            plate_text = ", ".join(plates)
                            additional_info.append(f"Rejestracja: {plate_text}")
                            label_text += f" ({plate_text})"
                        
                    elif category == 'traffic_signs':
                        text = text_extractor.extract_text(roi)
                        if text:
                            try:
                                text_label = text_clf.predict(text)
                                additional_info.append(f"Tekst: {text} ({text_label})")
                                label_text += f" - {text}"
                                
                                if text_label == "city":
                                    lat, lon = geolocate_text(text)
                                    if lat and lon:
                                        additional_info.append(f"Lokalizacja: {lat}, {lon}")
                            except:
                                additional_info.append(f"Tekst: {text}")
                        else:
                            additional_info.append("Znak bez tekstu")
                    
                    elif category == 'billboards':
                        text = detection.get('detected_text', '')
                        text_label = detection.get('text_classification', '')
                        
                        if text:
                            additional_info.append(f"Billboard: {text} ({text_label})")
                            label_text += f" - {text}"
                            
                            if text_label in ["city", "brand"]:
                                lat, lon = geolocate_text(text)
                                if lat and lon:
                                    additional_info.append(f"Lokalizacja: {lat}, {lon}")
                    
                    # Dodaj etykietę na obraz
                    cv2.putText(frame, label_text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, config['color'], 2)
                    
                    # Logowanie informacji
                    info_msg = f"[{frame_count}] {config['name']}: {class_name} (conf: {conf:.2f})"
                    for info in additional_info:
                        info_msg += f" | {info}"
                    
                    tqdm.write(info_msg)
                    frame_info.append(info_msg)

                # Zapisz ramkę do odpowiedniego katalogu
                if vehicle_detections or sign_detections or billboard_detections:
                    # Określ katalog docelowy
                    detection_types = []
                    if vehicle_detections:
                        detection_types.append('cars')
                    if sign_detections:
                        detection_types.append('signs')
                    if billboard_detections:
                        detection_types.append('billboards')
                    
                    if len(detection_types) == 1:
                        if detection_types[0] == 'cars':
                            output_dir = output_dirs['vehicles']
                            filename = f"car_frame_{frame_count}.jpg"
                        elif detection_types[0] == 'signs':
                            output_dir = output_dirs['traffic_signs']
                            filename = f"sign_frame_{frame_count}.jpg"
                        else:
                            output_dir = output_dirs['billboards']
                            filename = f"billboard_frame_{frame_count}.jpg"
                    else:
                        output_dir = output_dirs['mixed']
                        filename = f"mixed_frame_{frame_count}_{'_'.join(detection_types)}.jpg"
                    
                    output_path = output_dir / filename
                    # Save with high quality for JPEG
                    ext = os.path.splitext(str(output_path))[1].lower()
                    if ext == ".jpg" or ext == ".jpeg":
                        cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    elif ext == ".png":
                        cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    else:
                        cv2.imwrite(str(output_path), frame)
                    
                    tqdm.write(f"Frame {frame_count} saved to {output_path}")
                    
                    # AKTUALIZUJ PASEK Z INFORMACJĄ O ZAPISANIU
                    pbar.set_postfix(saved=str(output_path.name))

        finally:
            cap.release()
        
    logging.info("Video processing finished")
    logging.info(f"Total frames processed: {processed_frames}/{total_frames}")


if __name__ == "__main__":
    main()
