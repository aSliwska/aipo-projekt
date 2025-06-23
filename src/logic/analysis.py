import cv2
import sys
import logic.osm.open_street_maps as osm
from logic.osm.countries import all_countries
import logic.keywords.ekstrakcja_slow_kluczowych as keywords
import logic.lang_detect.language_detection as lang_detect
import logic.lang_detect.translate_phrases as translation
import logic.leftright.prawolewo as leftright
import logic.textdetection.deferred_glpd as deferred_text_detection
import logic.ocr.pipeline as ocr

def analyze_video(path_to_video, frame_skip):
    vid = cv2.VideoCapture(path_to_video)

    sys.argv = ['load_ufld', './logic/leftright/configs/culane.py']
    traffic_side_classifier = leftright.TrafficSideClassifier(path_appendix="./logic/leftright/")
    left_right_hits = {"Left-side traffic": 0, "Right-side traffic": 0, "Insufficient data": 0}

    countries_by_road_signs = [] 
    countries_by_language = {} 
    countries_by_car_license_plates = []
    regions_by_car_license_plates = []
    places_with_distances = {}

    ret, frame = vid.read()
    frame_idx = 0

    while (ret):
        if frame_idx % frame_skip == 0: # analyses every Nth frame, where N == frame_skip

            # traffic side detection
            traffic_side_classifier.set_image(frame)
            left_right_output = traffic_side_classifier.classify_traffic_side()
            left_right_hits[left_right_output] += 1

            # lightweight text detection (no API calls, no heavy processing)
            location_info = deferred_text_detection.analyze_frame_for_location(frame)
            
            # Process countries from road signs
            if location_info['sign_countries']:
                for country in location_info['sign_countries']:
                    if country not in countries_by_road_signs:
                        countries_by_road_signs.append(country)
            
            # Process countries and regions from license plates
            if location_info['plate_countries']:
                for country in location_info['plate_countries']:
                    if country not in countries_by_car_license_plates:
                        countries_by_car_license_plates.append(country)
            
            if location_info['plate_regions']:
                for region in location_info['plate_regions']:
                    if region not in regions_by_car_license_plates:
                        regions_by_car_license_plates.append(region)
            
            # Extract text for OCR processing (from billboard locations)
            ocr_output = location_info.get('billboard_locations', [])

            # language detection + translation
            if ocr_output:  # Only run if we have text to analyze
                language_detection_output = lang_detect.analyze_phrases(ocr_output, all_countries)
                countries_by_language = merge_dictionaries(language_detection_output, countries_by_language)
                translation_output = translation.translate_phrases(ocr_output)

                # keyword detection
                keyword_output = keywords.extract_named_entities_with_distance(translation_output, ocr_output)
                places_with_distances = merge_dictionaries(keyword_output, places_with_distances)

            try:
                image = frame.copy()
                detected_regions = [] #an array of images to process; hook the text extraction here

                if len(detected_regions): #many regions
                    ocr_output = {}
                    for region in detected_regions:
                        ocr_output_local = ocr.ocr_pipeline({"image" : image},just_text_region=True)['output']
                        for lang_group in ocr_output_local.keys(): #merge directories (in a different way)
                            ocr_output.setdefault(lang_group,[]).extend(ocr_output_local[lang_group])
                elif frame_idx % (frame_skip * 2) == 0: #extremely costly (out of proportion), shouldn't be run too often
                    ocr_output = ocr.ocr_pipeline({"image" : image})['output']
                    

                if ocr_output:  # the exact same stuff there
                    language_detection_output = lang_detect.analyze_phrases(ocr_output, all_countries)
                    countries_by_language = merge_dictionaries(language_detection_output, countries_by_language)
                    translation_output = translation.translate_phrases(ocr_output)

                    # keyword detection
                    keyword_output = keywords.extract_named_entities_with_distance(translation_output, ocr_output)
                    places_with_distances = merge_dictionaries(keyword_output, places_with_distances)
            except RuntimeError as e:
                print(e)



        ret, frame = vid.read()
        frame_idx += 1

    more_common_traffic_type = max(left_right_hits, key=left_right_hits.get)
    countries_by_road_side = traffic_side_classifier.fetch_countries(more_common_traffic_type)

    print("Finalizing text detection analysis...")
    final_results = deferred_text_detection.finalize_text_detection()

    # Update the detected countries with final results
    if final_results.get("plate_countries"):
        for country in final_results["plate_countries"]:
            if country not in countries_by_car_license_plates:
                countries_by_car_license_plates.append(country)
    
    if final_results.get("plate_regions"):
        for region in final_results["plate_regions"]:
            if region not in regions_by_car_license_plates:
                regions_by_car_license_plates.append(region)
    
    if final_results.get("sign_countries"):
        for country in final_results["sign_countries"]:
            if country not in countries_by_road_signs:
                countries_by_road_signs.append(country)

    print(f"Final detection results: {final_results['total_plates_detected']} plates from {final_results['total_frames_processed']} frames")
    print(f"Detected countries from plates: {final_results.get('plate_countries', [])}")
    print(f"Detected countries from signs: {final_results.get('sign_countries', [])}")
    print(countries_by_road_side, countries_by_road_signs, countries_by_car_license_plates, regions_by_car_license_plates, countries_by_language, places_with_distances)

    # return None, None
    return osm.predict_place(countries_by_road_side, countries_by_road_signs, countries_by_car_license_plates, 
        regions_by_car_license_plates, countries_by_language, places_with_distances)


def merge_dictionaries(x, y):
    z = x.copy()
    z.update(y) # y values overwrite x
    return z


if __name__ == "__main__":
    analyze_video('../../road_videos/1.mp4', 10)