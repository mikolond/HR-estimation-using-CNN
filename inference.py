from Models.estimator_model import Estimator
from Models.extractor_model import Extractor
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

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



class Inferencer:
    def __init__(self):
        self.extractor = Extractor()
        self.estimator = Estimator()
        self.extractor.eval()
        self.estimator.eval()
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print("Using GPU")
        else:
            print("Using CPU")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor.to(device)
        self.estimator.to(device)
        self.device = device

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
        
    def extract(self, faces):
        with torch.no_grad():
            x = torch.from_numpy(faces).permute(0, 3, 1, 2).float().to(self.device)
            output = self.extractor(x).detach().cpu().numpy()
        return output
    
    def estimate(self, sequence):
        with torch.no_grad():
            x = torch.tensor(sequence).float().to(self.device).reshape(1,1,300)
            output = self.estimator(x).detach().cpu().numpy()
        get_max_freq_padded(sequence.squeeze(), 30, output, output, pad_factor=10)
        return output * 60 # converting from Hz to bpm
    