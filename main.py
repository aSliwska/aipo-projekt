import cv2
import pytesseract
from ultralytics import YOLO
import requests
from urllib.parse import quote
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import logging
import subprocess
import json

# Configure logging for verbose output
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# ========== Klasyfikacja tekstu (SpaCy + Scikit-learn) ==========


class TextClassifier:
    def __init__(self):
        logging.debug('Initializing TextClassifier')
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression()
        self.trained = False

    def train(self, texts, labels):
        logging.info('Training TextClassifier with texts: %s', texts)
        X_train = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X_train, labels)
        self.trained = True
        logging.info('TextClassifier training complete')

    def predict(self, text):
        logging.debug('Predicting label for text: %s', text)
        if not self.trained:
            raise ValueError('Model nie został wytrenowany.')
        vect = self.vectorizer.transform([text])
        prediction = self.classifier.predict(vect)[0]
        logging.debug('Prediction result: %s', prediction)
        return prediction

# ========== Detekcja znaków drogowych (YOLOv5/v8) ==========


class RoadSignDetector:
    def __init__(self, model_path):
        logging.info('Loading YOLO model from: %s', model_path)
        self.model = YOLO(model_path)

    def detect(self, frame):
        logging.debug('Detecting road signs in frame')
        results = self.model(frame)
        boxes = []
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))
        logging.debug('Detected boxes: %s', boxes)
        return boxes

# ========== Klasyfikacja tablic rejestracyjnych (OpenALPR) ==========


class PlateRecognizer:
    def __init__(self, country="eu", alpr_path=None):
        logging.info('Initializing PlateRecognizer (CLI mode)')
        # Path to the alpr executable
        if alpr_path is None:
            # Default to build directory
            alpr_path = '/mnt/c/Users/wewek/Desktop/aipo/openalpr/src/build/alpr'
        self.alpr_path = alpr_path
        self.country = country

    def recognize(self, image_path):
        logging.debug('Recognizing plates in image (CLI): %s', image_path)
        try:
            result = subprocess.run([
                self.alpr_path,
                '-c', self.country,
                '-j',  # JSON output
                image_path
            ], capture_output=True, text=True, check=True)
            output = json.loads(result.stdout)
            plates = [plate['plate'] for plate in output.get('results', [])]
            logging.debug('Recognized plates: %s', plates)
            return plates
        except Exception as e:
            logging.error('Error running alpr CLI: %s', e)
            return []

    def __del__(self):
        pass  # No cleanup needed for CLI

# ========== Geolokalizacja przez Nominatim ==========


def geolocate_text(text):
    logging.info('Geolocating text: %s', text)
    query = quote(text)
    url = f"https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1"
    headers = {'User-Agent': 'GeoguessrApp/1.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                lat, lon = data[0]['lat'], data[0]['lon']
                logging.info('Geolocation result: %s, %s', lat, lon)
                return lat, lon
    except Exception as e:
        logging.error('Error during geolocation: %s', e)
    return None, None

# ========== Główna funkcja przetwarzająca wideo ==========


def main():
    logging.info('Starting main video processing')
    video_path = input('Podaj ścieżkę do pliku wideo: ').strip()
    if not os.path.isfile(video_path):
        logging.error('Nie znaleziono pliku wideo: %s', video_path)
        print('Nie znaleziono pliku wideo.')
        return

    # === Trening klasyfikatora tekstu ===
    text_clf = TextClassifier()
    train_texts = ["Warszawa", "60 km/h", "Euro", "Berlin", "Stop", "DUR", "USD"]
    train_labels = ["city", "sign", "currency", "city", "sign", "sign", "currency"]
    text_clf.train(train_texts, train_labels)

    # === Detekcja znaków ===
    sign_detector = RoadSignDetector("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error('Nie można otworzyć pliku wideo: %s', video_path)
        print('Nie można otworzyć pliku wideo.')
        return

    frame_count = 0
    plate_recognizer = PlateRecognizer()
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info('No more frames to read. Exiting loop.')
            break

        frame_count += 1
        if frame_count % 10 != 0:
            continue  # Pomijanie klatek dla wydajności

        logging.debug('Processing frame %d', frame_count)
        boxes = sign_detector.detect(frame)
        detection_in_frame = False
        detection_info = []

        for (x1, y1, x2, y2) in boxes:
            roi = frame[y1:y2, x1:x2]
            text = pytesseract.image_to_string(roi, lang='eng').strip()
            if text:
                label = text_clf.predict(text)
                log_msg = f"[{frame_count}] Wykryto tekst: '{text}' => Klasyfikacja: {label}"
                logging.info(log_msg)
                detection_in_frame = True
                detection_info.append(log_msg)
                if label == "city":
                    lat, lon = geolocate_text(text)
                    if lat and lon:
                        geo_msg = f"🗺️ Geolokalizacja: {text} => {lat}, {lon}"
                        logging.info(geo_msg)
                        detection_info.append(geo_msg)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Wykrywanie tablic rejestracyjnych (opcjonalnie dla każdej klatki)
        cv2.imwrite("temp_frame.jpg", frame)
        plates = plate_recognizer.recognize("temp_frame.jpg")
        for plate in plates:
            plate_msg = f"[{frame_count}] Zidentyfikowana tablica: {plate}"
            logging.info(plate_msg)
            detection_in_frame = True
            detection_info.append(plate_msg)

        # Save frame if anything detected
        if detection_in_frame:
            output_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(output_path, frame)
            logging.info(f"Frame {frame_count} saved to {output_path}")
            for info in detection_info:
                print(info)

    cap.release()
    logging.info('Video processing finished')


if __name__ == "__main__":
    main()
