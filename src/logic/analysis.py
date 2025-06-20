import cv2
import sys
import logic.osm.open_street_maps as osm
from logic.osm.countries import all_countries
import logic.keywords.ekstrakcja_slow_kluczowych as keywords
import logic.lang_detect.language_detection as lang_detect
import logic.lang_detect.translate_phrases as translation
import logic.leftright.prawolewo as leftright

def analyze_video(path_to_video, frame_skip):
    vid = cv2.VideoCapture(path_to_video)

    sys.argv = ['load_ufld', './logic/leftright/configs/culane.py']
    traffic_side_classifier = leftright.TrafficSideClassifier(path_appendix="./logic/leftright/")
    left_right_hits = {"Left-side traffic": 0, "Right-side traffic": 0}

    countries_by_road_signs = [] 
    countries_by_language = {} 
    countries_by_car_license_plates = []
    regions_by_car_license_plates = []
    places_with_distances = {}

    ret, frame = vid.read()
    frame_idx = 0

    ######################################################
    
    ######################################################

    while (ret):
        if frame_idx % frame_skip == 0: # analyses every Nth frame, where N == frame_skip

            traffic_side_classifier.set_image(frame)
            left_right_output = traffic_side_classifier.classify_traffic_side()
            left_right_hits[left_right_output] += 1


            # language_detection_output = lang_detect.analyze_phrases(ocr_output, all_countries)
            # countries_by_language = merge_dictionaries(language_detection_output, countries_by_language)
            # translation_output = translation.translate_phrases(ocr_output)

            # keyword_output = keywords.extract_named_entities_with_distance(translation_output, ocr_output)
            # places_with_distances = merge_dictionaries(keyword_output, places_with_distances)

        ret, frame = vid.read()
        frame_idx += 1

    more_common_traffic_type = max(left_right_hits, key=left_right_hits.get)
    countries_by_road_side = traffic_side_classifier.fetch_countries(more_common_traffic_type)

    # return None, None
    return osm.predict_place(countries_by_road_side, countries_by_road_signs, countries_by_car_license_plates, 
        regions_by_car_license_plates, countries_by_language, places_with_distances)


def merge_dictionaries(x, y):
    z = x.copy()
    z.update(y) # y values overwrite x
    return z


if __name__ == "__main__":
    analyze_video('../../road_videos/1.mp4', 10)