from extract_face import FaceExtractor
import numpy as np
import os
import cv2

class DatasetCreator:
    def __init__(self, path_in, path_out, sequence_length, model_path = 'mediapipe_model\\blaze_face_short_range.tflite', flag="ecg-fitness"):
        if not os.path.exists(path_in):
            raise FileNotFoundError(f"Input path {path_in} does not exist")
        self.path_in = path_in
        self.path_out = path_out
        self.N = sequence_length
        self.flag = flag
        self.face_extractor = FaceExtractor(model_path)
        self.cap = None
        self.actual_frame = 0
        self.actual_file = None
        self.fps = None
        self.frame_count = 0
        self.video_out_counter = 0

    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
    
    def get_video_specs(self):
        if self.cap is None:
            raise ValueError("Video not loaded")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def  load_frame(self):
        if self.cap is None:
            raise ValueError("Video not loaded")
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.actual_frame += 1
        return frame

    def unload_video(self):
        self.cap.release()
    
    def create_dataset(self):
        if self.flag == "ecg-fitness":
            self.create_dataset_ecg_fitness()


    def create_dataset_ecg_fitness(self):
        # init csv files
        c920 = None
        viatom = None
        if not os.path.exists(self.path_out):
            os.makedirs(self.path_out)
        subjects_list = os.listdir(self.path_in)
        for subject in subjects_list:
            folder_list = os.listdir(os.path.join(self.path_in, subject))
            for folder in folder_list:
                # create folder for video based on video out counter
                video_out_folder = os.path.join(self.path_out, f"video_{self.video_out_counter}")
                os.makedirs(video_out_folder)
                # load video c920-1.avi
                c920_path = os.path.join(self.path_in, subject, folder, 'c920-1.avi')
                self.load_video(c920_path)
                self.get_video_specs()
                for i in range(self.frame_count):
                    frame = self.load_frame()
                    # extract face from frame
                    face = self.face_extractor.extract_face(frame)




    

