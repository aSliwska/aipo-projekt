import cv2
import logging
import os
from typing import Dict, Any
from pathlib import Path
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Import our modular components
from logic.textdetection.text_classifier import TextClassifier
from logic.textdetection.text_extractor import TextExtractor
from logic.textdetection.object_detector import MultiObjectDetector
from logic.textdetection.plate_recognizer import PlateRecognizer
from logic.textdetection.plate_analyzer import PlateAnalyzer
from logic.textdetection.road_sign_analyzer import RoadSignAnalyzer

# Configure logging for verbose output
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


# ========== Main Processing Functions ==========
def process_video_frames(
    video_file,
    detector,
    plate_recognizer,
    text_extractor,
    text_clf,
    road_sign_analyzer,
    plate_analyzer,
    output_dirs,
    plates_txt_path,
    start_frame=0,
    frame_interval=10,
):
    """
    Processes video frames to detect vehicles, signs, billboards, recognize plates, annotate, and save frames.
    Uses process_frame for each frame.
    """
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        logging.error("Nie można otworzyć pliku wideo: %s", video_file)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info("Video info: %d frames, %.2f FPS", total_frames, fps)

    frame_count = 0
    processed_frames = 0
    seen_plates = set()

    with (
        tqdm(
            total=total_frames,
            desc="Processing video",
            unit="frames",
            position=0,
            leave=True,
        ) as pbar,
        logging_redirect_tqdm(),
    ):
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    tqdm.write("No more frames to read. Exiting loop.")
                    break

                frame_count += 1
                pbar.update(1)
                if frame_count < start_frame or frame_count % frame_interval != 0:
                    continue

                processed_frames += 1
                pbar.set_description(
                    f"Processing frame {frame_count}/{total_frames} (analyzed: {processed_frames})"
                )

                result = process_frame(
                    frame,
                    frame_count,
                    detector,
                    plate_recognizer,
                    text_extractor,
                    text_clf,
                    road_sign_analyzer,
                    plate_analyzer,
                    output_dirs,
                    plates_txt_path,
                    seen_plates,
                    save_to_file=True,
                )
                seen_plates = result["seen_plates"]
                output_path = result["output_path"]
                if output_path:
                    tqdm.write(f"Frame {frame_count} saved to {output_path}")
                    pbar.set_postfix(saved=str(output_path.name))
        finally:
            cap.release()
    logging.info("Video processing finished")
    logging.info("Total frames processed: %s/%s",
                 processed_frames, total_frames)
    return seen_plates, processed_frames, total_frames


def process_frame(
    frame,
    frame_count,
    detector,
    plate_recognizer,
    text_extractor,
    text_clf,
    road_sign_analyzer,
    plate_analyzer,
    output_dirs=None,
    plates_txt_path=None,
    seen_plates=None,
    save_to_file=False,
    outline_objects=True,
) -> Dict[str, Any]:
    """
    Processes a single frame to detect vehicles, signs, billboards, recognize plates, annotate, and optionally save the frame.
    Returns a dictionary with processed detections and outputs.
    """
    if seen_plates is None:
        seen_plates = set()

    detections = detector.detect(frame, text_extractor, text_clf)
    frame_info = []
    processed_detections = []

    # Kategoryzuj wykrycia w ramce
    vehicle_detections = []
    sign_detections = []
    billboard_detections = []

    for detection in detections:
        # Make a copy to avoid mutating the original
        det = dict(detection)
        if det["category"] == "vehicles":
            vehicle_detections.append(det)
        elif det["category"] == "traffic_signs":
            sign_detections.append(det)
        elif det["category"] == "billboards":
            billboard_detections.append(det)

    vehicles_with_plates = []
    for detection in vehicle_detections:
        x1, y1, x2, y2 = detection["bbox"]
        roi = frame[y1:y2, x1:x2]
        plates = plate_recognizer.recognize(roi)

        if plates:
            detection["plates"] = plates

            # Analyze plates for country/region information
            countries, regions = plate_analyzer.get_countries_for_plates(
                plates)
            if countries:
                detection["plate_countries"] = countries
            if regions:
                detection["plate_regions"] = regions

            vehicles_with_plates.append(detection)

            # Zapisz nowe tablice do pliku jeśli wymagane
            if save_to_file and plates_txt_path is not None:
                for plate in plates:
                    if plate not in seen_plates:
                        seen_plates.add(plate)
                        with open(plates_txt_path, "a", encoding="utf-8") as f:
                            f.write(f"{plate}\n")
                        logging.info("New plate detected: %s", plate)

    vehicle_detections = vehicles_with_plates

    all_detections = (
        vehicle_detections + sign_detections + billboard_detections
    )

    for detection in all_detections:
        x1, y1, x2, y2 = detection["bbox"]
        category = detection["category"]
        config = detection["config"]
        class_name = detection["class_name"]
        conf = detection["confidence"]
        roi = frame[y1:y2, x1:x2]
        label_text = f"{config['name']}: {class_name}"
        additional_info = []

        if category == "vehicles":
            plates = detection.get("plates", [])
            if plates:
                plate_text = ", ".join(plates)
                additional_info.append(f"Rejestracja: {plate_text}")
                label_text += f" ({plate_text})"

                # Add country/region information if available
                countries = detection.get("plate_countries", [])
                regions = detection.get("plate_regions", [])
                if countries:
                    additional_info.append(
                        f"Kraj tablicy: {', '.join(countries)}")
                if regions:
                    additional_info.append(
                        f"Region tablicy: {', '.join(regions)}")

        elif category == "traffic_signs":
            text = text_extractor.extract_text(roi)
            if text:
                try:
                    text_label = text_clf.predict(text)
                    additional_info.append(f"Tekst: {text} ({text_label})")
                    label_text += f" - {text}"

                    # Analyze sign for country-specific patterns
                    sign_countries = road_sign_analyzer.analyze_sign_text(text)
                    specific_country = road_sign_analyzer.is_country_specific_sign(
                        text)

                    if specific_country:
                        additional_info.append(
                            f"Kraj znaku: {specific_country}")
                        detection["sign_country"] = specific_country
                    elif sign_countries:
                        additional_info.append(
                            f"Możliwe kraje znaku: {', '.join(sign_countries)}")
                        detection["sign_countries"] = sign_countries

                    detection["text"] = text
                    detection["text_label"] = text_label
                except (ValueError, RuntimeError, AttributeError) as e:
                    logging.warning(
                        "Failed to classify text '%s': %s", text, e)
                    additional_info.append(f"Tekst: {text}")
                    detection["text"] = text
            else:
                additional_info.append("Znak bez tekstu")

        elif category == "billboards":
            text = detection.get("detected_text", "")
            text_label = detection.get("text_classification", "")
            if text:
                additional_info.append(f"Billboard: {text} ({text_label})")
                label_text += f" - {text}"

                # For billboards, we could also analyze for country-specific content
                if text_label in ["city", "brand"]:
                    # We would use the existing OSM service here instead of direct API calls
                    # For now, just log that this text could be geolocated
                    logging.info(
                        "Billboard text '%s' could be geolocated using OSM service", text)

        if outline_objects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), config["color"], 2)
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                config["color"],
                2,
            )
        info_msg = f"[{frame_count}] {config['name']}: {class_name} (conf: {conf:.2f})"
        for info in additional_info:
            info_msg += f" | {info}"
        logging.info(info_msg)
        frame_info.append(info_msg)
        processed_detections.append(dict(detection))
    output_path = None
    if save_to_file and (vehicle_detections or sign_detections or billboard_detections):
        detection_types = []
        if vehicle_detections:
            detection_types.append("cars")
        if sign_detections:
            detection_types.append("signs")
        if billboard_detections:
            detection_types.append("billboards")
        if len(detection_types) == 1:
            if detection_types[0] == "cars":
                output_dir = output_dirs["vehicles"]
                filename = f"car_frame_{frame_count}.jpg"
            elif detection_types[0] == "signs":
                output_dir = output_dirs["traffic_signs"]
                filename = f"sign_frame_{frame_count}.jpg"
            else:
                output_dir = output_dirs["billboards"]
                filename = f"billboard_frame_{frame_count}.jpg"
        else:
            output_dir = output_dirs["mixed"]
            filename = (
                f"mixed_frame_{frame_count}_{'_'.join(detection_types)}.jpg"
            )
        output_path = output_dir / filename
        ext = os.path.splitext(str(output_path))[1].lower()
        if ext == ".jpg" or ext == ".jpeg":
            cv2.imwrite(
                str(output_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100],
            )
        elif ext == ".png":
            cv2.imwrite(
                str(output_path),
                frame,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 0],
            )
        else:
            cv2.imwrite(str(output_path), frame)
        logging.info("Frame %s saved to %s", frame_count, output_path)
    return {
        "seen_plates": seen_plates,
        "output_path": output_path,
        "frame_info": frame_info,
        "frame": frame,
        "detections": processed_detections,
    }


def analyze_frame_for_location(frame):
    """
    Analyzes a single frame to detect location indicators (license plates, road signs, billboards)
    and returns a dictionary with countries/regions information.
    
    Args:
        frame: OpenCV image frame (numpy array)
        
    Returns:
        dict: Dictionary containing countries and regions information with structure:
        {
            'countries': [list of detected countries],
            'regions': [list of detected regions],
            'plate_countries': [countries from license plates],
            'plate_regions': [regions from license plates],
            'sign_countries': [countries from road signs],
            'billboard_locations': [locations from billboards]
        }
    """
    # Initialize components
    text_clf = TextClassifier()
    train_texts = [
        "Warszawa",
        "60 km/h",
        "Euro",
        "Berlin",
        "Stop",
        "DUR",
        "USD",
        "McDonald's",
        "Shell",
        "Coca-Cola",
    ]
    train_labels = [
        "city",
        "sign",
        "currency",
        "city",
        "sign",
        "sign",
        "currency",
        "brand",
        "brand",
        "brand",
    ]
    text_clf.train(train_texts, train_labels)

    detector = MultiObjectDetector("yolov8n.pt")
    plate_recognizer = PlateRecognizer()
    text_extractor = TextExtractor()
    road_sign_analyzer = RoadSignAnalyzer()
    plate_analyzer = PlateAnalyzer(use_api=False, use_glpd=True)  # Enable GLPD integration

    # Process the frame
    result = process_frame(
        frame,
        frame_count=1,
        detector=detector,
        plate_recognizer=plate_recognizer,
        text_extractor=text_extractor,
        text_clf=text_clf,
        road_sign_analyzer=road_sign_analyzer,
        plate_analyzer=plate_analyzer,
        save_to_file=False,
        outline_objects=False,
    )

    # Extract location information from detections
    location_info = {
        'countries': [],
        'regions': [],
        'plate_countries': [],
        'plate_regions': [],
        'sign_countries': [],
        'billboard_locations': []
    }

    for detection in result['detections']:
        # Extract plate countries and regions
        if 'plate_countries' in detection:
            location_info['plate_countries'].extend(detection['plate_countries'])
        if 'plate_regions' in detection:
            location_info['plate_regions'].extend(detection['plate_regions'])
        
        # Extract sign countries
        if 'sign_country' in detection:
            location_info['sign_countries'].append(detection['sign_country'])
        if 'sign_countries' in detection:
            location_info['sign_countries'].extend(detection['sign_countries'])
        
        # Extract billboard locations (cities, brands that could indicate location)
        if detection['category'] == 'billboards':
            text_label = detection.get('text_classification', '')
            text = detection.get('detected_text', '')
            if text_label in ['city', 'brand'] and text:
                location_info['billboard_locations'].append(text)

    # Remove duplicates and combine all countries/regions
    location_info['plate_countries'] = list(set(location_info['plate_countries']))
    location_info['plate_regions'] = list(set(location_info['plate_regions']))
    location_info['sign_countries'] = list(set(location_info['sign_countries']))
    location_info['billboard_locations'] = list(set(location_info['billboard_locations']))
    
    # Combine all countries and regions
    all_countries = (location_info['plate_countries'] + 
                    location_info['sign_countries'])
    all_regions = location_info['plate_regions']
    
    location_info['countries'] = list(set(all_countries))
    location_info['regions'] = list(set(all_regions))

    return location_info


def main():
    """
    Example usage of the analyze_frame_for_location function.
    Loads a sample frame and analyzes it for location information.
    """
    # Example: Load a sample frame
    sample_frame_path = Path("content/frame.jpg")
    if sample_frame_path.exists():
        frame = cv2.imread(str(sample_frame_path))
        if frame is not None:
            logging.info("Analyzing sample frame for location information...")
            location_info = analyze_frame_for_location(frame)
            
            logging.info("=== LOCATION ANALYSIS RESULTS ===")
            logging.info("All countries detected: %s", location_info['countries'])
            logging.info("All regions detected: %s", location_info['regions'])
            logging.info("Plate countries: %s", location_info['plate_countries'])
            logging.info("Plate regions: %s", location_info['plate_regions'])
            logging.info("Sign countries: %s", location_info['sign_countries'])
            logging.info("Billboard locations: %s", location_info['billboard_locations'])
            logging.info("=== END ANALYSIS ===")
            
            return location_info
        else:
            logging.error("Could not load sample frame: %s", sample_frame_path)
    else:
        logging.error("Sample frame not found: %s", sample_frame_path)
        logging.info("To use this module, call analyze_frame_for_location(frame) with your OpenCV frame")
    
    return None


if __name__ == "__main__":
    main()
