from Models.extractor_model import Extractor
from Loss.extractor_loss import ExtractorLoss
from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import datetime

N = 150 # length of the frame sequence
delta = 5/60 # offset from the true frequency
f_range = np.array([40, 240]) / 60 # all possible frequencies
sampling_f = 1/60 # sampling frequency in loss calculating




def get_max_freq(output,fps, hr):
    '''Use fourier transform to get the frequency with the highest amplitude and plots the frequency spectrum.
        other than the 0 HZ.
    '''
    output = output - np.mean(output)  # Remove DC component
    freqs = np.fft.fftfreq(len(output), d=1/fps)
    print("freqs", freqs)
    fft_values = np.fft.fft(output)
    fft_values = np.abs(fft_values) 
    
    # Ignore the zero frequency component
    fft_values[0] = 0
    

    valid_indices = (freqs > 40/60) & (freqs <= 240/60)
    freqs = freqs[valid_indices]
    print("freqs", freqs*60)
    fft_values = fft_values[valid_indices]
    max_freq_index = np.argmax(fft_values)
    max_freq = freqs[max_freq_index]
    # plot_sequence(output,freqs, fft_values, hr, "trash")
    
    return max_freq

def get_max_freq_padded(output, fps, hr, pad_factor=10): # Added pad_factor
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
    plot_sequence(output, freqs, fft_values, hr, "trash") # Different filename for padded plot

    return max_freq

def plot_sequence(sequence,freqs,fft, real_hr, save_path):
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
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.legend(["Frequency Spectrum", "Real HR"])
    plt.savefig(os.path.join(save_path, "frequency_spectrum.png"))
    plt.close()


def evaluate_dataset(dataset_loader, model, device, sequence_length = 150, batch_size=1, delta = 5/60, f_range = np.array([40, 240]) / 60, sampling_f = 1/60):
    dataset_loader.reset()
    augment_state = dataset_loader.augmentation
    dataset_loader.augmentation = False
    L2_list = []
    SNR_list = []
    dataset_done = False
    model.eval()
    print("Evaluating dataset")
    with torch.no_grad():
        while not dataset_done:
            sequence, f_true_list, fs_list, n_of_sequences, dataset_done = create_batch(dataset_loader, sequence_length, batch_size)
            if n_of_sequences == 0:
                break
            x = torch.tensor(sequence.reshape(n_of_sequences * sequence_length, 192, 128, 3).transpose(0, 3, 1, 2)).float().to(device)
            output = model(x).reshape(n_of_sequences, sequence_length)
            loss = ExtractorLoss().forward(output, torch.tensor(f_true_list).to(device), torch.tensor(fs_list).to(device), delta, sampling_f, f_range)
            SNR_list.append(loss.item())
            output_numpy_batch = output.detach().cpu().numpy().reshape(n_of_sequences * sequence_length)
            for i in range(n_of_sequences):
                output_numpy = output_numpy_batch[i*sequence_length:sequence_length*(i+1)]
                f_true = f_true_list[i]
                fs = fs_list[i]


                # evaluate L2 norm metric
                max_freq = get_max_freq_padded(output_numpy, fs, f_true)
                L2 = np.abs(max_freq - f_true) * 60 # convert to BPM
                L2_list.append(L2)
                progress = dataset_loader.progress()
                print("progress", int(progress[0]/progress[1]*100),"%", end="\r")
    dataset_loader.reset()
    dataset_loader.augmentation = augment_state

    return L2_list, SNR_list

def create_batch(dataset_loader, sequence_length, batch_size):
    sequence = np.zeros((batch_size, sequence_length, 192, 128, 3))
    f_true = np.zeros((batch_size))
    fs = np.zeros((batch_size))
    n_of_sequences = 0
    epoch_done = False
    for j in range(batch_size):
        cur_seq = dataset_loader.get_sequence()
        sequence[j] = cur_seq
        f_true[j] = dataset_loader.get_hr() / 60
        fs[j] = dataset_loader.get_fps()
        epoch_done = not dataset_loader.next_sequence()
        n_of_sequences = j + 1
    return sequence[:n_of_sequences], f_true[:n_of_sequences], fs[:n_of_sequences], n_of_sequences, epoch_done

def evaluate_weights(trn_dataset_loader, val_dataset_loader, weights_path, device, sequence_length = 150, batch_size=1, delta = 5/60, f_range = np.array([40, 240]) / 60, sampling_f = 1/60):
    model = Extractor().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    trn_L2_list, trn_SNR_list = evaluate_dataset(trn_dataset_loader, model, device, sequence_length, batch_size, delta, f_range, sampling_f)
    val_L2_list, val_SNR_list = evaluate_dataset(val_dataset_loader, model, device, sequence_length, batch_size, delta, f_range, sampling_f)
    trn_L2 = np.mean(trn_L2_list)
    trn_SNR = np.mean(trn_SNR_list)
    val_L2 = np.mean(val_L2_list)
    val_SNR = np.mean(val_SNR_list)
    return trn_L2, trn_SNR, val_L2, val_SNR

def evaluate_everything(trn_dataset_loader, val_dataset_loader, weights_folder_path, results_path, device, sequence_length = 150, batch_size=1, num_of_epochs = 10, delta = 5/60, f_range = np.array([40, 240]) / 60, sampling_f = 1/60):
    epochs_results = {"trn_L2": [], "trn_SNR": [], "val_L2": [], "val_SNR": []}
    for i in range(num_of_epochs + 1):
        weights_path = os.path.join(weights_folder_path,"model_epoch_" + str(i-1) + ".pth")
        trn_L2, trn_SNR, val_L2, val_SNR = evaluate_weights(trn_dataset_loader, val_dataset_loader, weights_path, device, sequence_length, batch_size, delta, f_range, sampling_f)
        epochs_results["trn_L2"].append(trn_L2)
        epochs_results["trn_SNR"].append(trn_SNR)
        epochs_results["val_L2"].append(val_L2)
        epochs_results["val_SNR"].append(val_SNR)
        write_csv(results_path, i, trn_L2, trn_SNR, val_L2, val_SNR)
    return epochs_results


def write_csv(results_path, epoch, trn_L2, trn_SNR, val_L2, val_SNR):
    csv_path = os.path.join(results_path, "evaluation_results.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "trn_L2", "trn_SNR", "val_L2", "val_SNR"])
        writer.writerow([epoch, trn_L2, trn_SNR, val_L2, val_SNR])


def save_results(epoch_results, results_path):
    csv_path = os.path.join(results_path, "evaluation_results.csv")
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "trn_L2", "trn_SNR", "val_L2", "val_SNR"])
        for i in range(len(epoch_results["trn_L2"])):
            writer.writerow([i, epoch_results["trn_L2"][i], epoch_results["trn_SNR"][i], epoch_results["val_L2"][i], epoch_results["val_SNR"][i]])
    print("Results saved to", csv_path)

def plot_results(epochs_results, results_path):
    trn_L2_list = []
    trn_SNR_list = []
    val_L2_list = []
    val_SNR_list = []
    for i in range(len(epochs_results["trn_L2"])):
        trn_L2_list.append(epochs_results["trn_L2"][i])
        trn_SNR_list.append(epochs_results["trn_SNR"][i])
        val_L2_list.append(epochs_results["val_L2"][i])
        val_SNR_list.append(epochs_results["val_SNR"][i])
    # 1 figure for L2
    plt.figure()
    plt.plot(trn_L2_list, label="Training L2")
    plt.plot(val_L2_list, label="Validation L2")
    plt.xlabel("Epoch")
    plt.ylabel("L2")
    plt.legend()
    plt.title("L2")
    plt.show()
    L2_save_path = os.path.join(results_path, "L2.png")
    plt.savefig(L2_save_path)
    # 1 figure for SNR
    plt.figure()
    plt.plot(trn_SNR_list, label="Training SNR")
    plt.plot(val_SNR_list, label="Validation SNR")
    plt.xlabel("Epoch")
    plt.ylabel("SNR")
    plt.legend()
    plt.title("SNR")
    plt.show()
    SNR_save_path = os.path.join(results_path, "SNR.png")
    plt.savefig(SNR_save_path)


if __name__ == "__main__":
    model = Extractor()
    device = torch.device("cuda:0")
    model.to(device)
    weights_path = "output/synthetic_weights/model_epoch_2.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    import yaml
    import csv
    config_data = yaml.safe_load(open("config_files/config_synthetic.yaml"))
    data = config_data["data"]
    optimizer = config_data["optimizer"]
    hr_data = config_data["hr_data"]
    train = config_data["train"]
    valid = config_data["valid"]
    # load data
    # weights_path = data["weights_dir"]
    benchmark_path = data["benchmark"]
    dataset_path = data["dataset_dir"]
    folders_path = os.path.join(dataset_path, "data.csv")
    folders = []
    with open(folders_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            folders.append(row)
    benchmark = yaml.safe_load(open(benchmark_path))
    train_folders = benchmark["trn"]
    valid_folders = benchmark["val"]
    train_videos_list = np.array([])
    valid_videos_list = np.array([])

    for idx in train_folders:
        train_videos_list = np.append(train_videos_list, np.array(folders[idx]))
    
    for idx in valid_folders:
        valid_videos_list = np.append(valid_videos_list, np.array(folders[idx]))

    # create training data loader
    train_sequence_length = 300
    train_shift = train["shift"]
    train_augment = train["augment"]
    train_data_loader = DatasetLoader(dataset_path, train_videos_list, N=300, step_size=300, augmentation=False)
    
    # create validation data loader
    valid_sequence_length = 300
    valid_shift = valid["shift"]
    valid_augment = valid["augment"]
    valid_data_loader = DatasetLoader(dataset_path, valid_videos_list, N=300, step_size=300, augmentation=False)

    trn_L2, trn_SNR, val_L2, val_SNR = evaluate_weights(train_data_loader, valid_data_loader, weights_path, device, train_sequence_length, 1, delta, f_range, sampling_f)
    print("Results:")
    print("trn_L2:", trn_L2)
    print("trn_SNR:", trn_SNR)
    print("val_L2:", val_L2)
    print("val_SNR:", val_SNR)

    # num_of_epochs = 30
    # batch_size = 1
    # import yaml
    # import csv
    # config_data = yaml.safe_load(open("config_files/config_extractor_median_1e-4_cum_10.yaml"))
    # data = config_data["data"]
    # optimizer = config_data["optimizer"]
    # hr_data = config_data["hr_data"]
    # train = config_data["train"]
    # valid = config_data["valid"]
    # # load data
    # weights_path = data["weights_dir"]
    # benchmark_path = data["benchmark"]
    # dataset_path = data["dataset_dir"]
    # folders_path = os.path.join(dataset_path, "data.csv")
    # folders = []
    # with open(folders_path, 'r') as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         folders.append(row)
    # benchmark = yaml.safe_load(open(benchmark_path))
    # train_folders = benchmark["trn"]
    # valid_folders = benchmark["val"]
    # train_videos_list = np.array([])
    # valid_videos_list = np.array([])

    # for idx in train_folders:
    #     train_videos_list = np.append(train_videos_list, np.array(folders[idx]))
    
    # for idx in valid_folders:
    #     valid_videos_list = np.append(valid_videos_list, np.array(folders[idx]))

    # # create training data loader
    # train_sequence_length = train["sequence_length"]
    # train_shift = train["shift"]
    # train_augment = train["augment"]
    # train_data_loader = DatasetLoader(dataset_path, train_videos_list, N=train_sequence_length, step_size=train_shift, augmentation=train_augment)
    
    # # create validation data loader
    # valid_sequence_length = valid["sequence_length"]
    # valid_shift = valid["shift"]
    # valid_augment = valid["augment"]
    # valid_data_loader = DatasetLoader(dataset_path, valid_videos_list, N=valid_sequence_length, step_size=valid_shift, augmentation=valid_augment)

    
    # # load HR data setting for loss function
    # delta = hr_data["delta"]/60
    # f_range = np.array(hr_data["frequency_range"])/60
    # sampling_f = hr_data["sampling_frequency"]/60
    # hr_data = {"delta": delta, "f_range": f_range, "sampling_f": sampling_f}

    # # device = input("Device to evaluate on: ")
    # # if not torch.cuda.is_available():
    # #     device = torch.device("cpu")
    # # else:
    # #     device = torch.device("cuda:" + device)
    # device = torch.device("cuda:0")

    # output_path = data["output_dir"]
    # # crate_folder with date and time in name in the output path
    # date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # results_path = os.path.join(output_path, date_time)
    # os.makedirs(results_path, exist_ok=True)


    # epochs_results = evaluate_everything(train_data_loader, valid_data_loader, weights_path, results_path, device, train_sequence_length, batch_size,num_of_epochs, delta, f_range, sampling_f)
    # print("Results:")
    # print(epochs_results)
    # # save_results(epochs_results, results_path)
    # plot_results(epochs_results, results_path)
