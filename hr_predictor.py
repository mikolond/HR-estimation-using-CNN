from model_estimator import Estimator
from model_extractor import Extractor
from face_extractor import FaceExtractor
import torch
import numpy as np
import os
import cv2


class HRPredictor:
    def __init__(self):
        self.extractor = Extractor()
        self.estimator = Estimator()
        self.extractor.eval()
        self.estimator.eval()
        self.face_extractor = FaceExtractor()
    
    def set_device(self, device):
        self.device = device
        self.extractor.to(device)
        self.estimator.to(device)
    
    def load_extractor_weights(self, model_path):
        if os.path.isfile(model_path):
            self.extractor.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Extractor {model_path} not found")
    
    def load_estimator_weights(self, model_path):
        if os.path.isfile(model_path):
            self.estimator.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Estimator {model_path} not found")
    
    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video {path} not found")
        
    def unload_video(self):
        self.cap.release()
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def get_face(self, image):
        return self.face_extractor.extract_face(image)
    
    def load_n_faces(self, n):
        faces = []
        for _ in range(n):
            frame = self.get_frame()
            if frame is None:
                break
            face = self.get_face(frame)
            if face is None:
                break
            faces.append(face)
        return faces
    
    def extract(self, faces):
        N = len(faces)
        with torch.no_grad():
            x = torch.from_numpy(np.array(faces).reshape(N,192,128,3)).permute(0, 3, 1, 2).float().to(self.device)
            output = self.extractor(x).detach().cpu().numpy()
        return output

    def estimate(self, sequence):
        with torch.no_grad():
            x = torch.tensor(sequence).float().to(self.device)
            output = self.estimator(x).detach().cpu().numpy()
        return output * 60 # converting from Hz to bpm
    
    def predict(self, faces):
        features = self.extract(faces)
        prediction = self.estimate(features)
        return prediction
    
    def process_video(self, video_path, sequence_length):
        self.load_video(video_path)
        predictions = []
        while True:
            faces = self.load_n_faces(sequence_length)
            if len(faces) == sequence_length:
                prediction = self.predict(faces)
                predictions.append(prediction)
            else:
                break
        self.unload_video()
        return predictions
    

if __name__ == '__main__':
    predictor = HRPredictor()
    predictor.set_device(torch.device('cuda'))
    extractor_weights_path = os.path.join("output","halmos_weights","latest_weights.pth")
    predictor.load_extractor_weights(extractor_weights_path)
    estimator_weights_path = os.path.join("output","estimator_weights","weights_ecg2.pth")
    predictor.load_estimator_weights(estimator_weights_path)
    video_path = 'video.mp4'
    predictions = predictor.process_video(video_path, 300)
    print(predictions)