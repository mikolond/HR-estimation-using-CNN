from model_estimator import Estimator
from model_extractor import Extractor
from face_extractor import FaceExtractor
import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


DEBUG = True


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
            # Optionally, save a sample face to verify the output
            cv2.imwrite("face.png", face)
            faces.append(face)
            print("Collected face with shape:", face.shape)
        faces = np.array(faces)
        print("Final faces array shape:", faces.shape)
        return faces
    
    def extract(self, faces):
        N = len(faces)
        print("N:",N)
        print("shape of faces:",faces.shape)
        with torch.no_grad():
            x = torch.from_numpy(faces).permute(0, 3, 1, 2).float().to(self.device)
            print("x shape:",x.shape)
            output = self.extractor(x).detach().cpu().numpy()
        return output

    def estimate(self, sequence):
        with torch.no_grad():
            print("sequence shape:",sequence.shape)
            x = torch.tensor(sequence).float().to(self.device).reshape(1,1,300)
            print("x shape:",x.shape)
            output = self.estimator(x).detach().cpu().numpy()
        return output * 60 # converting from Hz to bpm
    
    def predict(self, faces):
        features = self.extract(faces)
        prediction = self.estimate(features)
        return prediction
    
    def process_video(self, video_path, sequence_length):
        print("Processing video...")
        self.load_video(video_path)
        print("Video loaded")
        predictions = []
        while True:
            print("Loading faces...")
            faces = self.load_n_faces(sequence_length)
            if faces is None or len(faces) == 0:
                print("No faces found")
                break
            print("faces shape = ",faces.shape)
            print("Predicting...")
            if len(faces) == sequence_length:
                prediction = self.predict(faces)
                print("Prediction:", prediction)
                plt.plot(prediction)
                plt.savefig("prediction.png")
                plt.close()
                predictions.append(prediction)
            else:
                break
        self.unload_video()
        return predictions
    

if __name__ == '__main__':
    predictor = HRPredictor()
    predictor.set_device(torch.device('cuda'))
    extractor_weights_path = os.path.join("output","weights","extractor_weights_ecg.pth")
    predictor.load_extractor_weights(extractor_weights_path)
    estimator_weights_path = os.path.join("output","weights","estimator_weights_ecg.pth")
    predictor.load_estimator_weights(estimator_weights_path)
    video_path = "output_raw.avi"
    predictions = predictor.process_video(video_path, 300)
    print(predictions)