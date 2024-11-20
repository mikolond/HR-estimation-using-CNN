import cv2
import os
import numpy as np
from mtcnn import MTCNN

class DatasetCreator:
    def __init__(self, dataset_path, output_path, flag = 'ECG_fitness'):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.flag = flag
        
        self.detector = MTCNN()

    def load_video(self, path):
        # Load the video
        cap = cv2.VideoCapture(path)

        # Get the frames per second
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the width and height of frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return cap, fps, width, height


    def extract_face(self, cap):
        # Load the pre-trained model
