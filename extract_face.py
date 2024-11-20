import cv2
import os
import numpy as np
from mtcnn import MTCNN

class DatasetCreator:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.flag = 'ECG_fitness'

    def load_video(self, path):
        # Load the video
        cap = cv2.VideoCapture(path)

        # Get the frames per second
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the width and height of frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return cap, fps, width, height


    def extract_face(self):
        # Load the pre-trained model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Loop through the images in the dataset
        for image_name in os.listdir(self.dataset_path):
            image_path = os.path.join(self.dataset_path, image_name)

            # Read the image
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop through the faces
            for (x, y, w, h) in faces: