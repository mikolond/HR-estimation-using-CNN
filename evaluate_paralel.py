import os
import csv
import datetime
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from model_extractor import Extractor
from loss import ExtractorLoss
from dataset_loader import DatasetLoader
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Global constants
N = 150  # length of the frame sequence
delta = 5 / 60  # offset from the true frequency
f_range = np.array([40, 240]) / 60  # valid frequency range (converted)
sampling_f = 1 / 60  # sampling frequency in loss calculation

def get_max_freq(output, fps):
    """Compute the dominant frequency (ignoring DC) using a real FFT."""
    output_centered = output - np.mean(output)
    fft_values = np.abs(np.fft.rfft(output_centered))
    freqs = np.fft.rfftfreq(len(output), d=1 / (fps * 60))
    fft_values[0] = 0  # ignore DC
    valid_mask = (freqs > 40) & (freqs <= 240)
    if not np.any(valid_mask):
        return 0.0
    fft_values_valid = fft_values[valid_mask]
    freqs_valid = freqs[valid_mask]
    max_idx = np.argmax(fft_values_valid)
    return freqs_valid[max_idx]

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

def evaluate_dataset(dataset_loader, model, device, sequence_length=150, batch_size=1,
                     delta=5/60, f_range=np.array([40,240])/60, sampling_f=1/60):
    dataset_loader.reset()
    original_aug = dataset_loader.augmentation
    dataset_loader.augmentation = False
    L2_list = []
    SNR_list = []
    dataset_done = False
    model.eval()
    loss_fn = ExtractorLoss()
    batch_counter = 0

    try:
        with torch.no_grad():
            while not dataset_done:
                sequence, f_true_list, fs_list, n_of_sequences, dataset_done = create_batch(
                    dataset_loader, sequence_length, batch_size)
                if n_of_sequences == 0:
                    break

                seq_reshaped = sequence.reshape(n_of_sequences * sequence_length, 192, 128, 3).transpose(0, 3, 1, 2)
                x = torch.from_numpy(seq_reshaped).float().to(device)
                output = model(x).reshape(n_of_sequences, sequence_length)

                # Compute loss
                f_true_tensor = torch.from_numpy(f_true_list).to(device)
                fs_tensor = torch.from_numpy(fs_list).to(device)
                loss = loss_fn(output, f_true_tensor, fs_tensor, delta, sampling_f, f_range)
                SNR_list.append(-loss.item())

                # Process FFT for L2 metric
                output_batch = output.detach().cpu().numpy().reshape(n_of_sequences, sequence_length)
                f_true_array = np.array(f_true_list)
                fs_array = np.array(fs_list)

                if np.all(fs_array == fs_array[0]):
                    fps = fs_array[0]
                    output_centered = output_batch - np.mean(output_batch, axis=1, keepdims=True)
                    fft_values = np.abs(np.fft.rfft(output_centered, axis=1))
                    freqs = np.fft.rfftfreq(sequence_length, d=1 / (fps * 60))
                    fft_values[:, 0] = 0  # ignore DC
                    valid_mask = (freqs > 40) & (freqs <= 240)
                    if np.any(valid_mask):
                        fft_valid = fft_values[:, valid_mask]
                        freqs_valid = freqs[valid_mask]
                        max_indices = np.argmax(fft_valid, axis=1)
                        max_freqs = freqs_valid[max_indices]
                        L2_list.extend(np.abs(max_freqs - f_true_array))
                    else:
                        with ThreadPoolExecutor() as executor:
                            max_freqs = list(executor.map(lambda args: get_max_freq(*args),
                                                          zip(output_batch, fs_array)))
                        L2_list.extend(np.abs(np.array(max_freqs) - f_true_array))
                else:
                    with ThreadPoolExecutor() as executor:
                        max_freqs = list(executor.map(lambda args: get_max_freq(*args),
                                                      zip(output_batch, fs_array)))
                    L2_list.extend(np.abs(np.array(max_freqs) - f_true_array))

                batch_counter += 1
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        dataset_loader.reset()
        dataset_loader.augmentation = original_aug

    return L2_list, SNR_list

def evaluate_weights(model, trn_dataset_loader, val_dataset_loader, weights_path, device,
                     sequence_length=150, batch_size=1, delta=5/60,
                     f_range=np.array([40,240])/60, sampling_f=1/60):
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

def evaluate_epoch(args):
    """
    Evaluate one epoch (i.e. one weight file) independently.
    This function re-creates its own model and dataset loaders.
    """
    (epoch, weights_folder_path, config_file_path, device_str) = args
    device = torch.device(device_str)
    # Load configuration
    config_data = yaml.safe_load(open(config_file_path))
    data = config_data["data"]
    hr_data = config_data["hr_data"]
    train = config_data["train"]
    valid = config_data["valid"]
    
    # Re-create dataset loaders for this process
    dataset_path = data["dataset_dir"]
    folders_path = os.path.join(dataset_path, "data.csv")
    folders = []
    with open(folders_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            folders.append(row)
    benchmark = yaml.safe_load(open(data["benchmark"]))
    train_folders = benchmark["trn"]
    valid_folders = benchmark["val"]
    train_videos_list = []
    valid_videos_list = []
    for idx in train_folders:
        train_videos_list.extend(folders[idx])
    for idx in valid_folders:
        valid_videos_list.extend(folders[idx])
    train_videos_list = np.array(train_videos_list)
    valid_videos_list = np.array(valid_videos_list)
    
    train_sequence_length = train["sequence_length"]
    train_data_loader = DatasetLoader(dataset_path, train_videos_list,
                                      N=train_sequence_length, step_size=train["shift"],
                                      augmentation=train["augment"])
    valid_sequence_length = valid["sequence_length"]
    valid_data_loader = DatasetLoader(dataset_path, valid_videos_list,
                                      N=valid_sequence_length, step_size=valid["shift"],
                                      augmentation=valid["augment"])
    
    # Update HR data parameters
    delta_val = hr_data["delta"] / 60
    f_range_val = np.array(hr_data["frequency_range"]) / 60
    sampling_f_val = hr_data["sampling_frequency"] / 60
    
    weights_path = os.path.join(weights_folder_path, "model_epoch_" + str(epoch - 1) + ".pth")
    model = Extractor().to(device)
    results = evaluate_weights(model, train_data_loader, valid_data_loader,
                               weights_path, device, train_sequence_length, 1,
                               delta_val, f_range_val, sampling_f_val)
    # Return the epoch index and its evaluation results
    return (epoch, results)

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

def write_csv(results_path, epochs_results):
    csv_path = os.path.join(results_path, "evaluation_results.csv")
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "trn_L2", "trn_SNR", "val_L2", "val_SNR"])
        for epoch in sorted(epochs_results.keys()):
            trn_L2, trn_SNR, val_L2, val_SNR = epochs_results[epoch]
            writer.writerow([epoch, trn_L2, trn_SNR, val_L2, val_SNR])
    print("Results saved to", csv_path)

if __name__ == "__main__":
    # Set the multiprocessing start method to "spawn" to support CUDA.
    multiprocessing.set_start_method("spawn", force=True)
    
    num_of_epochs = 30
    batch_size = 1  # Example batch size for faster processing
    config_file_path = "config_files/config_extractor_synthetic.yaml"

    config_data = yaml.safe_load(open(config_file_path))
    data = config_data["data"]
    weights_folder_path = data["weights_dir"]
    output_path = data["output_dir"]
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Create a folder for results.
    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_path = os.path.join(output_path, date_time)
    os.makedirs(results_path, exist_ok=True)

    # Define a variable that sets how many weights are evaluated concurrently.
    max_parallel = 4  # Change this value as needed to control GPU memory usage.

    # Prepare arguments for each epoch evaluation.
    args_list = [(i, weights_folder_path, config_file_path, device_str) for i in range(num_of_epochs + 1)]
    
    # Use a multiprocessing Pool with a maximum of 'max_parallel' processes.
    with multiprocessing.Pool(processes=max_parallel) as pool:
        results = pool.map(evaluate_epoch, args_list)

    # Collect and sort results by epoch index.
    epochs_results = {"trn_L2": [], "trn_SNR": [], "val_L2": [], "val_SNR": []}
    for epoch, res in sorted(results):
        trn_L2, trn_SNR, val_L2, val_SNR = res
        epochs_results["trn_L2"].append(trn_L2)
        epochs_results["trn_SNR"].append(trn_SNR)
        epochs_results["val_L2"].append(val_L2)
        epochs_results["val_SNR"].append(val_SNR)
    
    print("Results:")
    print(epochs_results)
    write_csv(results_path, {i: res for i, res in sorted(results)})
    plot_results(epochs_results, results_path)
