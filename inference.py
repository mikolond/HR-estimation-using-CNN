import torch
from model import Extractor
from loss import ExtractorLoss
import numpy as np
from dataset_loader import DatasetLoader
import matplotlib.pyplot as plt
import numpy as np

N = 300 # length of the frame sequence
delta = 5/60 # offset from the true frequency
f_range = np.array([50, 240]) / 60 # all possible frequencies
sampling_f = 1/60 # sampling frequency in loss calculating

# MODEL_WEIGHTS = "model_second_try_lr_1e-4.pth"
MODEL_WEIGHTS = "model_weights_backup\\help_me_god.pth"
# MODEL_WEIGHTS = "model_weights\\model_epoch_1.pth"
# MODEL_WEIGHTS = "model_synthetic_N170.pth"
VISUALIZE = True

vis_count = 0

def get_max_freq(output,fps, hr):
    '''Use fourier transform to get the frequency with the highest amplitude and plots the frequency spectrum.
        other than the 0 HZ.
    '''
    global vis_count
    output = output - np.mean(output)  # Remove DC component
    freqs = np.fft.fftfreq(len(output), d=1/fps/60)
    fft_values = np.fft.fft(output)
    fft_values = np.abs(fft_values) 
    
    # Ignore the zero frequency component
    fft_values[0] = 0
    

    valid_indices = (freqs > 50) & (freqs <= 240)
    freqs = freqs[valid_indices]
    fft_values = fft_values[valid_indices]
    max_freq_index = np.argmax(fft_values)
    max_freq = freqs[max_freq_index]
    if VISUALIZE:
        
        plt.figure()
        plt.plot(freqs, fft_values)
        plt.scatter([hr], [0], color='red', label='HR')
        plt.legend()
        plt.xlim(40, 240)
        plt.title("Frequency Spectrum")
        plt.xlabel("Hr Frequency (Bpm)")
        plt.ylabel("Amplitude")
        plt.savefig("inference_vis\\freq_spectrum"+str(vis_count)+".png")
        plt.close()
        vis_count += 1
    
    return max_freq

def evaluate_dataset(dataset_loader, model, device):
    L2_list = []
    SNR_list = []
    dataset_done = False
    model.eval()
    while not dataset_done:
        frames = dataset_loader.get_sequence()
        hr = dataset_loader.get_hr()
        fps = dataset_loader.get_fps()
        x = torch.tensor(frames.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
        with torch.no_grad():
            output = model(x).reshape(N)
        output_numpy = output.detach().cpu().numpy()
        # evaluate L2 norm metric
        max_freq = get_max_freq(output_numpy, fps, hr)
        L2 = np.abs(max_freq - hr)
        L2_list.append(L2)
        # evaluate snr metric (using loss function)
        f_true = torch.tensor([hr / 60], dtype=torch.float32).to(device)
        fs = torch.tensor([fps], dtype=torch.float32).to(device)
        loss = ExtractorLoss().forward(output.reshape(1,N), f_true, fs, delta, sampling_f, f_range)
        SNR_list.append(-loss.item())
        dataset_done = not dataset_loader.next_sequence()
        progress = dataset_loader.progress()
        print("progress", progress)

    return L2_list, SNR_list
    


class ExtractorInference:
    def __init__(self, model_path, data_loader, device):
        self.model_path = model_path
        self.data_loader = data_loader
        self.device = device
        self.model = Extractor().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def infer(self):
        for i in range(800):
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
    import os
    valid_path = os.path.join("C:\projects\dataset_creator_test_output","valid_dataset")
    valid_videos_list = os.listdir(valid_path)
    valid_data_loader = DatasetLoader(valid_path, valid_videos_list, N=N, step_size=N)
    valid_videos_list = []
    for i in range(179, 192):
        valid_videos_list.append("video_" + str(i))
    loader = DatasetLoader(valid_path, valid_videos_list, N=N, step_size=N)

    # loader = DatasetLoader("C:\\projects\\dataset_creator_test_output", videos, N=N, step_size=N)
    device = torch.device("cuda")
    print("device", device)
    model = Extractor().to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    l2_list, snr_list = evaluate_dataset(loader, model, device)
    # print("L2 list", l2_list)
    # print("SNR list", snr_list)
    print("mean average error:", np.mean(l2_list))
    print("root mean square error:", np.sqrt(np.mean(np.array(l2_list)**2)))
    print("SNR mean:", np.mean(snr_list))



