import torch
from model_estimator import Estimator
from estimator_dataset_loader import EstimatorDatasetLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import time

class EstimatorEval:
    def __init__(self, weights_path, device):
        self.model = Estimator().to(device)
        self.model.eval()
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.device = device


    def infer(self, sequence):
        sequence = sequence.reshape(1,150,1).transpose(0,2,1)
        x = torch.tensor(sequence).float().to(self.device)
        output = self.model(x)
        return output.item()
    
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
    plt.plot(freqs,fft)
    # plot the real hr as a dot on the graph with y axis value of 0
    plt.scatter(real_hr, 0, color='red')
    plt.scatter(predicted, 0, color='green')
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.savefig(os.path.join(save_path, "frequency_spectrum.png"))
    plt.close()



if __name__ == "__main__":
    weights_path = os.path.join("output","estimator_weights","weights_exp1.pth")
    device = torch.device("cuda:0")
    dataset_path = os.path.join("datasets", "estimator_synthetic")
    train_videos_list = ["video_150.csv"]

    data_loader = EstimatorDatasetLoader(dataset_path, train_videos_list, N=150, step_size=50)

    estimator = EstimatorEval(weights_path,device)

    for i in range(20):
        sequence, real_hr = data_loader.get_sequence()
        # print("sequence:",sequence)
        predicted_hr = estimator.infer(sequence)
        fig1 = plt.figure()
        get_max_freq_padded(sequence, 30, real_hr/60, predicted_hr, pad_factor=10)
        
        print(f"predicted hr:{predicted_hr}, real hr:{real_hr/60}")
        data_loader.next_sequence()
        time.sleep(1)


