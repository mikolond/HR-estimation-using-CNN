import torch
from model import Extractor
import numpy as np
from dataset_loader import DatasetLoader
import matplotlib.pyplot as plt
import numpy as np

N = 200 # length of the frame sequence

# MODEL_WEIGHTS = "model_second_try_lr_1e-4.pth"
# MODEL_WEIGHTS = "model_first_try_lr_1e-4.pth"
MODEL_WEIGHTS = "model_weights\\model_epoch_4.pth"
# MODEL_WEIGHTS = "model.pth"
# MODEL_WEIGHTS = "model_synthetic_N170.pth"

def get_max_freq(output,fps, hr):
    '''Use fourier transform to get the frequency with the highest amplitude and plots the frequency spectrum.
        other than the 0 HZ.
    '''
    output = output - np.mean(output)  # Remove DC component
    freqs = np.fft.fftfreq(len(output), d=1/fps/60)
    fft_values = np.fft.fft(output)
    fft_values = np.abs(fft_values) 
    
    # Ignore the zero frequency component
    fft_values[0] = 0
    
    max_freq_index = np.argmax(fft_values)
    max_freq = freqs[max_freq_index]
    
    plt.figure()
    plt.plot(freqs, fft_values)
    plt.scatter([hr], [0], color='red', label='HR')
    plt.legend()
    plt.xlim(20, 220)
    plt.title("Frequency Spectrum")
    plt.xlabel("Hr Frequency (Bpm)")
    plt.ylabel("Amplitude")
    plt.show()
    
    return max_freq



class ExtractorInference:
    def __init__(self, model_path, data_loader, device):
        self.model_path = model_path
        self.data_loader = data_loader
        self.device = device
        self.model = Extractor().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def infer(self):
        for i in range(10):
            self.data_loader.next_sequence()
        frames = self.data_loader.get_sequence()
        fps = self.data_loader.get_fps()
        hr = self.data_loader.get_hr()
        x = torch.tensor(frames.transpose(0, 3, 1, 2), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(x).reshape(N)
        output_numpy = output.detach().cpu().numpy()
        print("output shape", output.shape)
        self.plot_output(output_numpy)
        print("max freq", get_max_freq(output_numpy, fps, hr), "real freq", hr)

    def plot_output(self, output_numpy):
        plt.figure()
        plt.plot(output_numpy)
        plt.title("Output")
        plt.show()

if __name__ == "__main__":
    loader = DatasetLoader("C:\\projects\\dataset_creator_test_output", ["video_5"], N=N)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    inference = ExtractorInference(MODEL_WEIGHTS, loader, device)
    inference.infer()


