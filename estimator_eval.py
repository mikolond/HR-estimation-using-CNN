import torch
from Models.estimator_model import Estimator
from Models.extractor_model import Extractor
from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import time
plot_counter = 0

class EstimatorEval:
    def __init__(self, extractor_model, extractor_weights_path, estimator_weights_path, device, N):
        self.model = Estimator().to(device)
        self.extractor_model = extractor_model.to(device)
        self.extractor_model.load_state_dict(torch.load(extractor_weights_path, map_location=device))
        self.extractor_model.eval()
        self.model.eval()
        self.model.load_state_dict(torch.load(estimator_weights_path, map_location=device))
        self.device = device
        self.N = N


    def infer(self, sequence):
        sequence = sequence.reshape(1,self.N,1).transpose(0,2,1)
        # print("sequence shape:",sequence.shape)
        x = torch.tensor(sequence).float().to(self.device)
        output = self.model(x)
        return output.item()
    
    def get_average_deviation(self, data_loader):
        '''Calculate the average deviation between average of the data in loader and the real data in loader.'''
        hr_data = np.array([])
        dataset_done = False
        while not dataset_done:
            _, real_hr = data_loader.get_sequence()
            hr_data = np.append(hr_data, real_hr)
            dataset_done = not data_loader.next_sequence()
        average_hr = np.mean(hr_data)
        average_deviation = np.mean(np.abs(hr_data - average_hr))
        return average_deviation
    
    def evaluate(self, trn_loader, val_loader):
        loss = {}
        print("evaluation training loss")
        loss["trn_rmse"], loss["trn_mae"] = self.validate(trn_loader)
        print("evaluation validation loss")
        loss["val_rmse"], loss["val_mae"] = self.validate(val_loader)
        return loss
    
    def infer_extractor(self, sequence):
        with torch.no_grad():
            x = torch.tensor(sequence.reshape(self.N, 192, 128, 3).transpose(0, 3, 1, 2)).float().to(self.device)
            output = self.extractor_model(x).reshape(self.N).cpu().numpy()
        return output
    
    def validate(self, data_loader):
        self.model.eval()
        errors = []
        epoch_done = False
        with torch.no_grad():
            while not epoch_done:
                sequence = data_loader.get_sequence()
                hr_data = data_loader.get_hr()
                extractor_output = self.infer_extractor(sequence)
                # print("extractor_output shape:",extractor_output.shape)
                # print("extractor_output:",extractor_output)
                predicted = self.infer(extractor_output) * 60

                loss = predicted - hr_data
                errors.append(loss)
                epoch_done = not data_loader.next_sequence()
                # get max freq
                get_max_freq_padded(extractor_output, 30, hr_data/60, predicted/60, pad_factor=10)
        errors = np.array(errors)
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        return rmse, mae
        
    
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
    plot_sequence(output, freqs, fft_values, hr,predicted, "trash/with_learning") # Different filename for padded plot
    time.sleep(0.5)

    return max_freq

def plot_sequence(sequence,freqs,fft, real_hr,predicted, save_path):
    global plot_counter
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
    plt.legend(["Frequency","Real HR", "Predicted HR"])
    if plot_counter <= 50:
        plt.savefig(os.path.join(save_path, "frequency_spectrum"+str(plot_counter)+".png"))
        plot_counter += 1
    plt.close()



if __name__ == "__main__":
    import yaml
    import csv
    config_data = yaml.safe_load(open("config_files/config_debug_synthetic.yaml"))
    data = config_data["data"]
    config_data = config_data["estimator"]
    optimizer = config_data["optimizer"]
    train = config_data["train"]
    valid = config_data["valid"]
    # load data
    weights_path = data["weights_dir"]
    extractor_weights_path = os.path.join(weights_path, "best_extractor_weights.pth")
    if not os.path.exists(extractor_weights_path):
        raise Exception("Extractor weights not found")
    estimator_weights_path = os.path.join(weights_path, "best_estimator_weights.pth")
    if not os.path.exists(estimator_weights_path):
        raise Exception("Estimator weights not found")
    benchmark_path = data["benchmark"]
    dataset_path = data["dataset_dir"]
    output_path = data["output_dir"]
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
    train_sequence_length = train["sequence_length"]
    train_shift = train["shift"]
    train_augment = train["augment"]
    train_data_loader = DatasetLoader(dataset_path, train_videos_list, N=train_sequence_length, step_size=train_shift, augmentation=train_augment)
    
    # create validation data loader
    valid_sequence_length = valid["sequence_length"]
    valid_shift = valid["shift"]
    valid_augment = valid["augment"]
    valid_data_loader = DatasetLoader(dataset_path, valid_videos_list, N=valid_sequence_length, step_size=valid_shift, augmentation=valid_augment)


    sequence_length = train["sequence_length"]

    # load data for training
    lr = float(optimizer["lr"])
    batch_size = optimizer["batch_size"]
    num_epochs = optimizer["max_epochs"]
    patience = optimizer["patience"]
    decrease_lr = optimizer["decrease_lr"]
    lr_decay = optimizer["lr_decay"]
    lr_decay_epochs = optimizer["lr_decay_epochs"]

    device = input("Device to train on: ")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + device)
    # device = torch.device("cpu")
    extractor_model = Extractor()
    extractor_model.load_state_dict(torch.load(extractor_weights_path))

    evaluator = EstimatorEval(extractor_model, extractor_weights_path,estimator_weights_path, device, 300)
    evaluator.evaluate(train_data_loader, valid_data_loader)
    # weights_path = os.path.join("output","estimator_weights","best_model.pth")

    # import csv
    # import yaml
    # dataset_path = os.path.join("datasets", "estimator_ecg_fitness_latest")
    # folders_path = os.path.join(dataset_path, "data.csv")
    # folders = []
    # with open(folders_path, 'r') as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         folders.append(row)

    # benchmark_path = os.path.join("benchmarks", "benchmark_ecg.yaml")
    # benchmark = yaml.safe_load(open(benchmark_path))
    # train_folders = benchmark["trn"]
    # valid_folders = benchmark["val"]
    # train_videos_list = np.array([])
    # valid_videos_list = np.array([])

    # for idx in train_folders:
    #     train_videos_list = np.append(train_videos_list, np.array(folders[idx]))
    
    # for idx in valid_folders:
    #     valid_videos_list = np.append(valid_videos_list, np.array(folders[idx]))

    # # add .csv after every video name
    # for i in range(len(train_videos_list)):
    #     train_videos_list[i] += ".csv"
    # for i in range(len(valid_videos_list)):
    #     valid_videos_list[i] += ".csv"
    # data_loader = EstimatorDatasetLoader(dataset_path, train_videos_list, N=300, step_size=300)

    # # create training data loader
    # train_data_loader = EstimatorDatasetLoader(dataset_path, train_videos_list, N=300, step_size=300)
    
    # # create validation data loader
    # valid_data_loader = EstimatorDatasetLoader(dataset_path, valid_videos_list, N=300, step_size=300)

    # device = torch.device("cuda:0")

    # estimator = EstimatorEval(weights_path,device, 300)

    # loss = estimator.evaluate(train_data_loader, valid_data_loader)
    # print(loss)
    # train_data_loader.reset()
    # valid_data_loader.reset()
    # average_trn_deviation = estimator.get_average_deviation(train_data_loader)
    # average_val_deviation = estimator.get_average_deviation(valid_data_loader)
    # print("average training deviation:", average_trn_deviation)
    # print("average validation deviation:", average_val_deviation)

    # for i in range(20):
    #     sequence = data_loader.get_sequence()
    #     real_hr = data_loader.get_hr()
    #     # print("sequence:",sequence)
    #     predicted_hr = estimator.infer(sequence)
    #     fig1 = plt.figure()
    #     get_max_freq_padded(sequence, 30, real_hr/60, predicted_hr, pad_factor=10)
        
    #     print(f"predicted hr:{predicted_hr}, real hr:{real_hr/60}")
    #     data_loader.next_sequence()
    #     time.sleep(0.5)


