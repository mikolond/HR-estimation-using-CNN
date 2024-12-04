from extract_face import FaceExtractor
import numpy as np
import os
import cv2

class DatasetCreator:
    def __init__(self, path_in, path_out, sequence_length, model_path = 'mediapipe_model\\blaze_face_short_range.tflite', flag="ecg-fitness"):
        self.path_in = path_in
        self.path_out = path_out
        self.N = sequence_length
        self.flag = flag
        self.face_extractor = FaceExtractor(model_path)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    
    def create_dataset(self):
        if not os.path.exists(self.path_in):
            raise FileNotFoundError(f"Input path {self.path_in} does not exist")
        if not os.path.exists(self.path_out):
            os.mkdir(self.path_out)
        for file in os.listdir(self.path_in):


    

