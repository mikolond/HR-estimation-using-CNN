from model_extractor import Extractor
# from my_extractor import Extractor
from loss import ExtractorLoss
from dataset_loader import DatasetLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import datetime
import yaml

# Global constants
N = 150  # length of the frame sequence
delta = 5 / 60  # offset from the true frequency
f_range = np.array([40, 240]) / 60  # valid frequency range in BPM (converted to Hz in minutes)
sampling_f = 1 / 60  # sampling frequency in loss calculation

def get_max_freq(output,fps, hr):
    '''Use fourier transform to get the frequency with the highest amplitude and plots the frequency spectrum.
        other than the 0 HZ.
    '''
    output = output - np.mean(output)  # Remove DC component
    freqs = np.fft.fftfreq(len(output), d=1/fps)
    fft_values = np.fft.fft(output)
    fft_values = np.abs(fft_values) 
    
    # Ignore the zero frequency component
    fft_values[0] = 0
    

    valid_indices = (freqs > 40/60) & (freqs <= 240/60)
    freqs = freqs[valid_indices]
    fft_values = fft_values[valid_indices]
    max_freq_index = np.argmax(fft_values)
    max_freq = freqs[max_freq_index]
    # plot_sequence(output,freqs, fft_values, hr, "trash")
    
    return max_freq

def evaluate_dataset(dataset_loader, model, device, sequence_length=150, batch_size=1,
                     delta=5/60, f_range=np.array([40, 240])/60, sampling_f=1/60):
    dataset_loader.reset()
    original_aug = dataset_loader.augmentation
    dataset_loader.augmentation = False
    L2_list = []
    SNR_list = []
    dataset_done = False
    model.eval()
    loss_fn = ExtractorLoss()  # Instantiate loss function once
    batch_counter = 0

    try:
        with torch.no_grad():
            while not dataset_done:
                sequence, f_true_list, fs_list, n_of_sequences, dataset_done = create_batch(dataset_loader, sequence_length, batch_size)
                if n_of_sequences == 0:
                    break

                # Convert and reshape the batch using torch.from_numpy for speed
                seq_reshaped = sequence[:n_of_sequences].reshape(n_of_sequences * sequence_length, 192, 128, 3).transpose(0, 3, 1, 2)
                x = torch.from_numpy(seq_reshaped).float().to(device)
                output = model(x).reshape(n_of_sequences, sequence_length)

                # Compute loss
                f_true_tensor = torch.from_numpy(f_true_list[:n_of_sequences]).to(device)
                fs_tensor = torch.from_numpy(fs_list[:n_of_sequences]).to(device)
                loss = loss_fn(output, f_true_tensor, fs_tensor, delta, sampling_f, f_range)
                SNR_list.append(loss.item())

                # Reshape outputs for FFT processing
                output_batch = output.detach().cpu().numpy().reshape(n_of_sequences, sequence_length)
                f_true_array = np.array(f_true_list[:n_of_sequences])
                fs_array = np.array(fs_list[:n_of_sequences])

                # If all fps values in the batch are the same, vectorize the FFT computation
                if np.all(fs_array == fs_array[0]):
                    fps = fs_array[0]
                    output_centered = output_batch - np.mean(output_batch, axis=1, keepdims=True)
                    fft_values = np.abs(np.fft.rfft(output_centered, axis=1))
                    freqs = np.fft.rfftfreq(sequence_length, d=1 /fps )
                    fft_values[:, 0] = 0  # ignore DC
                    valid_mask = (freqs > 40/60) & (freqs <= 240/60)
                    if np.any(valid_mask):
                        fft_valid = fft_values[:, valid_mask]
                        freqs_valid = freqs[valid_mask]
                        max_indices = np.argmax(fft_valid, axis=1)
                        max_freqs = freqs_valid[max_indices]
                        L2_list.extend(np.abs(max_freqs - f_true_array))
                    else:
                        for i in range(n_of_sequences):
                            max_freq = get_max_freq(output_batch[i], fs_array[i])
                            L2_list.append(np.abs(max_freq - f_true_array[i])*60)
                else:
                    for i in range(n_of_sequences):
                        max_freq = get_max_freq(output_batch[i], fs_array[i])
                        L2_list.append(np.abs(max_freq - f_true_array[i])*60)

                batch_counter += 1
                if batch_counter % 10 == 0:
                    progress = dataset_loader.progress()
                    print("progress", int(progress[0] / progress[1] * 100), "%", end="\r")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        dataset_loader.reset()
        dataset_loader.augmentation = original_aug

    return L2_list, SNR_list

def create_batch(dataset_loader, sequence_length, batch_size):
    sequence = np.empty((batch_size, sequence_length, 192, 128, 3), dtype=np.float32)
    f_true = np.empty((batch_size), dtype=np.float32)
    fs = np.empty((batch_size), dtype=np.float32)
    n_of_sequences = 0
    epoch_done = False
    for j in range(batch_size):
        cur_seq = dataset_loader.get_sequence()
        sequence[j] = cur_seq
        f_true[j] = dataset_loader.get_hr() / 60
        fs[j] = dataset_loader.get_fps()
        n_of_sequences = j + 1
        epoch_done = not dataset_loader.next_sequence()
        if epoch_done:
            break
    return sequence[:n_of_sequences], f_true[:n_of_sequences], fs[:n_of_sequences], n_of_sequences, epoch_done

def evaluate_weights(model, trn_dataset_loader, val_dataset_loader, weights_path, device,
                     sequence_length=150, batch_size=1, delta=5/60,
                     f_range=np.array([40, 240])/60, sampling_f=1/60):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    trn_L2_list, trn_SNR_list = evaluate_dataset(trn_dataset_loader, model, device,
                                                 sequence_length, batch_size, delta, f_range, sampling_f)
    val_L2_list, val_SNR_list = evaluate_dataset(val_dataset_loader, model, device,
                                                 sequence_length, batch_size, delta, f_range, sampling_f)
    trn_L2 = np.mean(trn_L2_list) if trn_L2_list else float('nan')
    trn_SNR = np.mean(trn_SNR_list) if trn_SNR_list else float('nan')
    val_L2 = np.mean(val_L2_list) if val_L2_list else float('nan')
    val_SNR = np.mean(val_SNR_list) if val_SNR_list else float('nan')
    return trn_L2, trn_SNR, val_L2, val_SNR

def evaluate_everything(trn_dataset_loader, val_dataset_loader, weights_folder_path, results_path, device,
                        sequence_length=150, batch_size=1, num_of_epochs=10, delta=5/60,
                        f_range=np.array([40, 240])/60, sampling_f=1/60):
    epochs_results = {"trn_L2": [], "trn_SNR": [], "val_L2": [], "val_SNR": []}
    # Instantiate model once and reuse it for all epochs
    model = Extractor().to(device)
    for i in range(num_of_epochs + 1):
        weights_path = os.path.join(weights_folder_path, "model_epoch_" + str(i - 1) + ".pth")
        trn_L2, trn_SNR, val_L2, val_SNR = evaluate_weights(model, trn_dataset_loader, val_dataset_loader,
                                                              weights_path, device, sequence_length,
                                                              batch_size, delta, f_range, sampling_f)
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
            writer.writerow([i, epoch_results["trn_L2"][i], epoch_results["trn_SNR"][i],
                             epoch_results["val_L2"][i], epoch_results["val_SNR"][i]])
    print("Results saved to", csv_path)

def plot_results(epochs_results, results_path):
    epochs = np.arange(len(epochs_results["trn_L2"]))
    # Plot L2 results
    plt.figure()
    plt.plot(epochs, epochs_results["trn_L2"], label="Training L2")
    plt.plot(epochs, epochs_results["val_L2"], label="Validation L2")
    plt.xlabel("Epoch")
    plt.ylabel("L2")
    plt.legend()
    plt.title("L2")
    L2_save_path = os.path.join(results_path, "L2.png")
    plt.savefig(L2_save_path)
    plt.close()

    # Plot SNR results
    plt.figure()
    plt.plot(epochs, epochs_results["trn_SNR"], label="Training SNR")
    plt.plot(epochs, epochs_results["val_SNR"], label="Validation SNR")
    plt.xlabel("Epoch")
    plt.ylabel("SNR")
    plt.legend()
    plt.title("SNR")
    SNR_save_path = os.path.join(results_path, "SNR.png")
    plt.savefig(SNR_save_path)
    plt.close()

if __name__ == "__main__":
    num_of_epochs = 3
    batch_size = 1
    config_data = yaml.safe_load(open("config_files/config_extractor.yaml"))
    data = config_data["data"]
    hr_data = config_data["hr_data"]
    train = config_data["train"]
    valid = config_data["valid"]

    weights_folder_path = data["weights_dir"]
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

    # Build video lists efficiently
    train_videos_list = []
    valid_videos_list = []
    for idx in train_folders:
        train_videos_list.extend(folders[idx])
    for idx in valid_folders:
        valid_videos_list.extend(folders[idx])
    train_videos_list = np.array(train_videos_list)
    valid_videos_list = np.array(valid_videos_list)

    # Create training and validation data loaders
    train_sequence_length = train["sequence_length"]
    train_shift = train["shift"]
    train_augment = train["augment"]
    train_data_loader = DatasetLoader(dataset_path, train_videos_list,
                                      N=train_sequence_length, step_size=train_shift, augmentation=train_augment)

    valid_sequence_length = valid["sequence_length"]
    valid_shift = valid["shift"]
    valid_augment = valid["augment"]
    valid_data_loader = DatasetLoader(dataset_path, valid_videos_list,
                                      N=valid_sequence_length, step_size=valid_shift, augmentation=valid_augment)

    # Update HR data settings for the loss function
    delta = hr_data["delta"] / 60
    f_range = np.array(hr_data["frequency_range"]) / 60
    sampling_f = hr_data["sampling_frequency"] / 60

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_path = data["output_dir"]
    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_path = os.path.join(output_path, date_time)
    os.makedirs(results_path, exist_ok=True)

    epochs_results = evaluate_everything(train_data_loader, valid_data_loader, weights_folder_path,
                                         results_path, device, train_sequence_length,
                                         batch_size, num_of_epochs, delta, f_range, sampling_f)
    print("Results:")
    print(epochs_results)
    plot_results(epochs_results, results_path)
