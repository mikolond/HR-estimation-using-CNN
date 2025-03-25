from Models.estimator_model import Estimator
from Models.extractor_model import Extractor
from face_extractor import FaceExtractor
import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


DEBUG = True

def get_max_freq_padded(output, fps, hr,predicted, pad_factor=10): # Added pad_factor
    '''Use fourier transform to get the frequency with the highest amplitude with zero-padding.

    Args:
        output (np.array): The input signal.
        fps (float): Sampling rate (frames per second).
        hr (str): Description for plot title.
        pad_factor (int): Factor by which to increase signal length through padding.
                          e.g., pad_factor=10 means padded length is 10 times original.

    Returns:
        float: The frequency with the highest amplitude (in Hz).
    '''
    output = output - np.mean(output)  # Remove DC component
    original_length = len(output)
    padded_length = original_length * pad_factor # Calculate padded length
    padding = np.zeros(padded_length - original_length) # Create zero padding
    output_padded = np.concatenate((output, padding)) # Apply padding

    freqs = np.fft.fftfreq(padded_length, d=1/fps) # Use padded length for freqs
    # print("freqs (padded)", freqs)
    fft_values = np.fft.fft(output_padded)
    fft_values = np.abs(fft_values)

    # Ignore the zero frequency component
    fft_values[0] = 0

    valid_indices = (freqs > 40/60) & (freqs <= 240/60)
    freqs = freqs[valid_indices]
    # print("freqs (padded, BPM)", freqs * 60)
    fft_values = fft_values[valid_indices]

    max_freq_index = np.argmax(fft_values)
    max_freq = freqs[max_freq_index]
    plot_sequence(output, freqs, fft_values, hr,predicted, "trash") # Different filename for padded plot

    return max_freq

def plot_sequence(sequence,freqs,fft, real_hr,predicted, save_path):
    plt.figure()
    plt.plot(sequence)
    plt.title("Sequence")
    plt.xlabel("Frame")
    plt.ylabel("Amplitude")
    plt.savefig(os.path.join(save_path, "sequence.png"))
    plt.close()
    plt.figure()
    plt.plot(freqs*60,fft)
    # plot the real hr as a dot on the graph with y axis value of 0
    plt.scatter(real_hr*60, 0, color='red')
    plt.scatter(predicted*60, 0, color='green')
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency [bpm]")
    plt.ylabel("Amplitude")
    plt.legend(["Frequency","Real HR", "Predicted HR"])
    plt.savefig(os.path.join(save_path, "frequency_spectrum.png"))
    plt.close()




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
            get_max_freq_padded(sequence.squeeze(), 30, output, output, pad_factor=10)
        return output * 60 # converting from Hz to bpm
    
    def predict(self, faces):
        features = self.extract(faces)
        print("features shape:",features.shape)
        plt.plot(features.squeeze())
        plt.savefig("features.png")
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
                predictions.append(prediction)
            else:
                break
        self.unload_video()
        return predictions
    

if __name__ == '__main__':
    predictor = HRPredictor()
    predictor.set_device(torch.device('cuda'))
    extractor_weights_path = os.path.join("output","weights","model_epoch_35.pth")
    predictor.load_extractor_weights(extractor_weights_path)
    estimator_weights_path = os.path.join("output","estimator_weights","best_model.pth")
    predictor.load_estimator_weights(estimator_weights_path)
    video_path = "test_videos/me_70_phone.mp4"
    predictions = predictor.process_video(video_path, 300)
    print(predictions)