from Models.estimator_model2 import Estimator
from Models.extractor_model4 import Extractor
from face_extractor import FaceExtractor
from utils import load_model_class
from inference import Inferencer
import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import yaml


DEBUG = True
CONFIG_PATH = os.path.join("config_files", "config_process_video.yaml")

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
    def __init__(self,extractor_path=None, estimator_path=None):
        self.inferencer = Inferencer(extractor_path=extractor_path, estimator_path=estimator_path)
        self.face_extractor = FaceExtractor()

    def set_device(self, device):
        self.inferencer.set_device(device)
    
    def load_extractor_weights(self, weights_path):
        self.inferencer.load_extractor_weights(weights_path)
    
    def load_estimator_weights(self, weights_path):
        self.inferencer.load_estimator_weights(weights_path)
    
    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video {path} not found")
    
    def get_frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def unload_video(self):
        self.cap.release()
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def get_face(self, image):
        # Rotate the image 90 degrees anti-clockwise if width is larger than height (needs testing can differ for different video formats)
        return self.face_extractor.extract_face(image)
    
    def load_n_faces(self, n):
        faces = []
        i = 0
        while True:
            frame = self.get_frame()
            if frame is None:
                break
            face = self.get_face(frame)
            if face is None and i == 0:
                i = -1
            faces.append(face)
            # save face to file
            cv2.imwrite("face.png", face)
            i += 1
            if i >= n:
                break
        faces = np.array(faces)
        return faces
    
    def extract(self, faces):
        return self.inferencer.extract(faces)

    def estimate(self, sequence, batch_size=1):
        return self.inferencer.estimate(sequence, batch_size=batch_size)
    
    def predict(self, faces):
        rppg = self.extract(faces)
        prediction = self.estimate(rppg)
        return prediction
    
    def process_video(self, video_path, sequence_length):
        print("Processing video...")
        self.load_video(video_path)
        frame_count = self.get_frame_count()
        print("Video loaded, extracting rppg signal...")
        rppg = np.array([])
        faces = np.array([])
        last_faces = np.array([])
        while True:
            last_faces = faces
            faces = self.load_n_faces(sequence_length)
            if faces is None or len(faces) == 0:
                break
            if len(faces) == sequence_length:
                current_rppg = self.extract(faces)
                rppg = np.append(rppg, current_rppg, axis=0)
            elif len(faces) > 0:
                current_length = len(faces)
                complement_length = sequence_length - current_length
                current_faces = np.concatenate((faces, last_faces[-complement_length:]), axis=0)
                current_rppg = self.extract(current_faces)
                current_rppg = current_rppg[-current_length:]
                rppg = np.concatenate((rppg, current_rppg), axis=0)
                break
        # estimate extrated rppg signal
        print("RPPG signal extracted, estimating HR...")
        predictions = np.array([])
        if len(rppg) < sequence_length:
            print("Not enough frames to estimate HR")
            return []
        estimates = np.array([])
        estimates_count = frame_count - sequence_length
        rppg_batch = []
        batch_count = 0
        batch_max = 600
        for i in range(estimates_count):
            rppg_batch.append(rppg[i:i+sequence_length])
            batch_count += 1
            if batch_count == batch_max:
                rppg_batch = np.array(rppg_batch)
                current_estimate = self.estimate(rppg_batch, batch_size=batch_max)
                estimates = np.append(estimates, current_estimate, axis=0)
                rppg_batch = []
                batch_count = 0
        if len(rppg_batch) > 0:
            rppg_batch = np.array(rppg_batch)
            current_estimate = self.estimate(rppg_batch, batch_size=len(rppg_batch))
            estimates = np.append(estimates,current_estimate, axis=0)
        
        for i in range(sequence_length//2):
            predictions = np.append(predictions, estimates[0])
        predictions = np.append(predictions, estimates)
        for i in range(sequence_length//2):
            predictions = np.append(predictions, estimates[-1])
        

        self.unload_video()
        return predictions
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a video and predict HR (offline demo)")
    parser.add_argument("config_path", type=str, help="Path to the config file", default=None)
    parser.add_argument("video_path", type=str, help="Path to the video file", default=None)
    parser.add_argument("output_path", type=str, help="Path to the output file", default=None)
    args = parser.parse_args()
    if args.config_path is None:
        raise Exception("No config path provided")
    else:
        if not os.path.exists(args.config_path):
            raise Exception("Config path does not exist")
        else:
            config_path = args.config_path
    if args.video_path is None:
        raise Exception("No video path provided")
    else:
        if not os.path.exists(args.video_path):
            raise Exception("Video path does not exist")
        else:
            video_path = args.video_path

    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    # load models
    models = config_data["models"]
    extractor_path = models["extractor_model"]
    estimator_path = models["estimator_model"]
    predictor = HRPredictor(extractor_path=extractor_path, estimator_path=estimator_path)

    # load weights
    weights = config_data["weights"]
    if not os.path.exists(weights["extractor_weights"]):
        raise FileNotFoundError(f"Extractor weights {weights['extractor']} not found")
    if not os.path.exists(weights["estimator_weights"]):
        raise FileNotFoundError(f"Estimator weights {weights['estimator']} not found")
    predictor.load_extractor_weights(weights["extractor_weights"])
    predictor.load_estimator_weights(weights["estimator_weights"])

    # load video and output path
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    device = input("Device to use: ")
    if not torch.cuda.is_available() or device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + device)

    predictor.set_device(device)

    predictions = predictor.process_video(video_path, 300)
    video_name = os.path.basename(video_path)
    # change to .txt file
    file_name = os.path.splitext(video_name)[0] + ".txt"
    output_file_path = os.path.join(output_path, file_name)
    with open(output_file_path, 'w') as f:
        for prediction in predictions:
            f.write(str(prediction) + "\n")
    print("Predictions saved to", output_file_path)