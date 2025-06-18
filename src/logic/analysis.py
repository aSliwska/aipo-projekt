import osm.open_street_maps as osm
import cv2

def analyze_video(path_to_video):
    vid = cv2.VideoCapture(path_to_video)

    countries_by_road_side = []
    countries_by_road_signs = [] 
    countries_by_language = [] 
    countries_by_car_license_plates = []
    regions_by_car_license_plates = []
    places_with_distances = {}

    ret, frame = vid.read()
    i = 0
    while (ret):
        # TODO: analyze the frame
        ret, frame = vid.read()
        i+=1
    print(i)

    return osm.predict_place(countries_by_road_side, countries_by_road_signs, countries_by_car_license_plates, 
        regions_by_car_license_plates, countries_by_language, places_with_distances)

if __name__ == "__main__":
    analyze_video('../../road_videos/1.mp4')