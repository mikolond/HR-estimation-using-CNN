from inference import ExtractorInference
from model import Extractor
from loss import ExtractorLoss
from dataset_loader import DatasetLoader
import torch
import matplotlib.pyplot as plt
import numpy as np

N = 300 # length of the frame sequence
delta = 5/60 # offset from the true frequency
f_range = np.array([45, 240]) / 60 # all possible frequencies
sampling_f = 1/60 # sampling frequency in loss calculating


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
    

    valid_indices = (freqs > 50) & (freqs <= 240)
    freqs = freqs[valid_indices]
    fft_values = fft_values[valid_indices]
    max_freq_index = np.argmax(fft_values)
    max_freq = freqs[max_freq_index]
    
    return max_freq

def evaluate_dataset(dataset_loader, model, device):
    L2_list = []
    SNR_list = []
    dataset_done = False
    model.eval()
    print("Evaluating dataset")
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
        print("progress", int(progress[0]/progress[1]*100),"%", end="\r")

    return L2_list, SNR_list

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("dataset_folder_path", type=str, help="Path to the dataset")
    parser.add_argument("--device", type=str, help="Device to train on", default="cuda:0")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results", default=False)

    args = parser.parse_args()
    model_path = args.model_path
    dataset_folder_path = args.dataset_folder_path
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = Extractor().to(device)
    model.load_state_dict(torch.load(model_path))
    dataset_loader = DatasetLoader(dataset_folder_path, None, N = N, step_size=N)
    L2_list, SNR_list = evaluate_dataset(dataset_loader, model, device)
    print("Mean average error: ", np.mean(L2_list))
    print("Root mean square error:", np.sqrt(np.mean(np.array(L2_list)**2)))
    print("Mean SNR: ", np.mean(SNR_list))
    if args.visualize:
        plt.figure()
        plt.plot(L2_list)
        plt.title("L2 norm")
        plt.show()

