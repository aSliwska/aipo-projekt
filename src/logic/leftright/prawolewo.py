import os
import sys
import cv2
import random
import torch
import yaml
import numpy as np
import joblib

from ultralytics import YOLO
from logic.leftright.model.model import parsingNet
from logic.leftright.utils.common import merge_config
from logic.leftright.data.constant import culane_row_anchor

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

#=========================WAŻNE======================
# Linijka poniżej ma być wywołana przed importowaniem modułu bo merge_config() jest ********y tutaj
#                           |
# sys.argv = ['load_ufld', './configs/culane.py'] 
#====================================================

# Użycie:
#  classifier = TrafficSideClassifier(img, path_appendix="./")
#  classifier.get_possible_countries()

# =========================INPUT==================================== POSTAC: image, ew. path_appendix do zmiany ścieżki jeżeli jest wywoływany gdzie indziej

#=========================OUTPUT====================================: String: "Left-side traffic"/"right-side traffic"/"Insufficient data", Lista: Possible countries: ['Afghanistan', 'Albania', 'Algeria', ..., 'Zimbabwe'], może być pusta



class TrafficSideClassifier:    
    def __init__(self, img=None, path_appendix=""):
        self.img = img
        self.path_appendix = path_appendix

        self.clf = joblib.load(self.path_appendix+"traffic_side_classifier.pkl")
        self.countries = self.load_countries()
        self.yolo = YOLO(self.path_appendix+"yolov8n.pt")
        self.ufld = self.load_ufld()

    def load_ufld(self):
        cfg_path = self.path_appendix+'configs/culane.py'
        args, cfg = merge_config()
        
        if cfg.dataset == 'CULane':
            cls_num_per_lane = 18
        elif cfg.dataset == 'Tusimple':
            cls_num_per_lane = 56
        else:
            raise NotImplementedError

        net = parsingNet(
            pretrained=False,
            backbone=cfg.backbone,
            cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
            use_aux=False
        ).cuda()

        state_dict = torch.load(self.path_appendix+'checkpoint/culane_18.pth')
        if 'model' in state_dict:
            state_dict = state_dict['model']

        net.load_state_dict(state_dict)
        net.eval()
        return net

    def load_countries(self):  
        with open(self.path_appendix+"countries.yaml", "r") as file:
            data = yaml.safe_load(file)
        return data
    
    def fetch_countries(self, traffic_side):
        if traffic_side == "Right-side traffic":
            return self.countries.get("RHT", [])
        elif traffic_side == "Left-side traffic":
            return self.countries.get("LHT", [])
        else:
            return []    
    def detect_lane_and_vehicles(self):
        img = self.img
        ori_img = img.copy()

        img_ufld = cv2.resize(img, (800, 288))
        img_ufld = img_ufld / 255.0
        img_ufld = img_ufld.transpose(2, 0, 1)
        img_ufld = torch.FloatTensor(img_ufld).unsqueeze(0).cuda()

        with torch.no_grad():
            out = self.ufld(img_ufld)[0].cpu().numpy()

        lanes = []
        for i in range(4):
            lane = []
            for j in range(out.shape[1]):
                idx = np.argmax(out[i, j])
                if idx != out.shape[2] - 1:
                    x = int(idx * 800 / out.shape[2])
                    y = int(culane_row_anchor[j] * ori_img.shape[0] / 288)
                    lane.append((x, y))
            if lane:
                lanes.append(lane)

        results = self.yolo(img)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        car_boxes = [box for box, cls in zip(boxes, classes) if int(cls) == 2]

        return lanes, car_boxes, ori_img

    def classify_traffic_side(self):
        lanes, cars, img = self.detect_lane_and_vehicles()

        if not lanes or not cars:
            return "Insufficient data"

        lane_x_positions = [x for lane in lanes for x, _ in lane]
        if not lane_x_positions:
            return "Insufficient data"

        lane_center = np.median(lane_x_positions)
        img_width = img.shape[1]

        car_x = [(box[0] + box[2]) / 2 for box in cars]
        left = sum([x < lane_center for x in car_x])
        right = sum([x >= lane_center for x in car_x])

        features = [
            left, right,
            lane_center / img_width,
            np.mean(car_x) / img_width if car_x else 0,
            len(lanes),
            len(cars)
        ]

        prediction = self.clf.predict([features])[0]
        return "Left-side traffic" if prediction == 1 else "Right-side traffic"

    def set_image(self, img):
        self.img = img
    
    def get_possible_countries(self):
        traffic_type = self.classify_traffic_side()
        return traffic_type, self.fetch_countries(traffic_type)



if __name__ == "__main__":
    sys.argv = ['load_ufld', './logic/leftright/configs/culane.py'] 

    classifier = TrafficSideClassifier(img=cv2.imread('logic/content/frame.jpg'), path_appendix="./logic/")
    traffic_type, possible_countries = classifier.get_possible_countries()

    print("Traffic type:", traffic_type)
    print("Possible countries:", possible_countries)