import torch
from Models.estimator_model import Estimator
from Models.extractor_model import Extractor
# from Models.extractor_latent import Extractor
from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
from Datasets_handlers.Estimator.dataset_loader import EstimatorDatasetLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import time
plot_counter = 0

def get_statistics(data_loader):
    '''Calculate the average deviation between average of the data in loader and the real data in loader.'''
    statistics = {}
    hr_data = np.array([])
    dataset_done = False
    while not dataset_done:
        real_hr = data_loader.get_hr_list()

        next_out = data_loader.next_sequence()
        if next_out is None:
            dataset_done = True
        else:
            real_hr = real_hr[len(real_hr)-next_out:]
        hr_data = np.append(hr_data, real_hr)
        progress = data_loader.get_progress()
        print(f"Progress: {progress[0]}/{progress[1]}", end="\r")

    hr_data = np.array(hr_data)
    average = np.mean(hr_data)
    std_deviation = np.std(hr_data)
    mean_deviation = np.mean(np.abs(hr_data - average))
    statistics["mean_deviation"] = mean_deviation
    statistics["average"] = average
    statistics["std_deviation"] = std_deviation
    return statistics

class EstimatorEval:
    def __init__(self, extractor_model, extractor_weights_path, estimator_weights_path, device, N, output_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
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
    

    
    def evaluate(self, data_loader, tag = "unknown"):
        loss = {}
        # print("evaluating dataset")
        loss["rmse"], loss["mae"], loss["pearson"] = self.validate(data_loader, tag)
        return loss
    
    def infer_extractor(self, sequence):
        with torch.no_grad():
            x = torch.tensor(sequence.reshape(self.N, 192, 128, 3).transpose(0, 3, 1, 2)).float().to(self.device)
            output = self.extractor_model(x).reshape(self.N).cpu().numpy()
        return output
    
    def make_predictions(self, data_loader):
        self.model.eval()
        ground_truth = []
        predicted = []
        data_loader.reset()
        epoch_done = False
        with torch.no_grad():
            while not epoch_done:
                sequence = data_loader.get_sequence()
                hr_data = data_loader.get_hr()

                extractor_output = self.infer_extractor(sequence)
                # print("extractor_output shape:",extractor_output.shape)
                # print("extractor_output:",extractor_output)
                prediction = self.infer(extractor_output) * 60
                epoch_done = not data_loader.next_sequence()
                ground_truth.append(hr_data)
                predicted.append(prediction)
                progress = data_loader.get_progress()
                print(f"Progress: {progress[0]}/{progress[1]}", end="\r")
                get_max_freq_padded(extractor_output, 30, hr_data, prediction, pad_factor=10)

        return ground_truth, predicted
    
    def validate(self, data_loader, tag):
        ground_truth, predicted = self.make_predictions(data_loader)
        plot_pearson(ground_truth, predicted, os.path.join(self.output_path), tag)
        errors = np.array(predicted) - np.array(ground_truth)
        # computes the rmse
        rmse = np.sqrt(np.mean(errors**2))
        # computes the mae
        mae = np.mean(np.abs(errors))
        # computes the pearsion correlation
        pearson = np.corrcoef(ground_truth, predicted)[0, 1]
        return rmse, mae, pearson
        
    
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

def plot_pearson(ground_truth, predicted, save_path, tag):
    plt.figure()
    pearson = np.corrcoef(ground_truth, predicted)[0, 1]
    minimum = min(min(ground_truth), min(predicted))
    maximum = max(max(ground_truth), max(predicted))
    plt.xlim(minimum, maximum)
    plt.ylim(minimum, maximum)
    plt.plot([minimum, maximum], [minimum, maximum], 'r--')
    plt.xticks(np.arange(minimum, maximum, 5),rotation=45)
    plt.yticks(np.arange(minimum, maximum, 5))
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(ground_truth, predicted)
    plt.title(f"{tag}-set, Pearson correlation R={pearson:.2f}")
    plt.xlabel("Ground Truth bpm")
    plt.ylabel("Predicted bpm")
    namefile = "pearson_correlation_" + tag + ".png"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, namefile), bbox_inches='tight')
    plt.clf()
    plt.close()

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
    freqs = freqs * 60
    plt.plot(freqs,fft)
    # plot the real hr as a dot on the graph with y axis value of 0
    plt.scatter(real_hr, 0, color='red')
    plt.scatter(predicted, 0, color='green')
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.legend(["Frequency","Real HR", "Predicted HR"])
    # if plot_counter <= 50:
    #     plt.savefig(os.path.join(save_path, "frequency_spectrum"+str(plot_counter)+".png"))
    #     plot_counter += 1
    plt.savefig(os.path.join(save_path, "frequency_spectrum.png"))
    plt.close()



if __name__ == "__main__":
    import yaml
    import csv
    config_data = yaml.safe_load(open("config_files/pure_local/config_eval_test.yaml"))
    data = config_data["data"]
    weights = config_data["weights"]
    extractor_weights_path = weights["extractor_weights"]
    estimator_weights_path = weights["estimator_weights"]
    if not os.path.exists(extractor_weights_path):
        raise Exception("Extractor weights path does not exist")
    if not os.path.exists(estimator_weights_path):
        raise Exception("Estimator weights path does not exist")
    
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
    test_folders = benchmark["tst"]
    train_videos_list = np.array([])
    valid_videos_list = np.array([])
    test_videos_list = np.array([])

    for idx in train_folders:
        train_videos_list = np.append(train_videos_list, np.array(folders[idx]))
    
    for idx in valid_folders:
        valid_videos_list = np.append(valid_videos_list, np.array(folders[idx]))
    
    for idx in test_folders:
        test_videos_list = np.append(test_videos_list, np.array(folders[idx]))

    dataset_options = config_data["dataset_options"]
    seq_length = dataset_options["sequence_length"]
    step_size = dataset_options["shift"]
    # create training data loader
    train_data_loader = DatasetLoader(dataset_path, train_videos_list, N=seq_length, step_size=step_size, augmentation=False)
    
    # create validation data loader
    valid_data_loader = DatasetLoader(dataset_path, valid_videos_list, N=seq_length, step_size=step_size, augmentation=False)

    # create test data loader
    test_data_loader = DatasetLoader(dataset_path, test_videos_list, N=seq_length, step_size=step_size, augmentation=False)


    device = input("Device to train on: ")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + device)
    # device = torch.device("cpu")
    extractor_model = Extractor()
    extractor_model.load_state_dict(torch.load(extractor_weights_path, map_location=device))

    evaluator = EstimatorEval(extractor_model, extractor_weights_path,estimator_weights_path, device, seq_length, output_path)
    print("evaluating train data")
    loss = evaluator.evaluate(train_data_loader, tag = "train")
    print("train loss:", loss)
    print("evaluating validation data")
    loss = evaluator.evaluate(valid_data_loader, tag = "validation")
    print("validation loss:", loss)
    print("evaluating test data")
    loss = evaluator.evaluate(test_data_loader, tag = "test")
    print("test loss:", loss)

    train_data_loader.reset()
    valid_data_loader.reset()
    test_data_loader.reset()

    dataloader_statistics = get_statistics(train_data_loader)
    print("train data loader statistics:", dataloader_statistics)
    dataloader_statistics = get_statistics(valid_data_loader)
    print("validation data loader statistics:", dataloader_statistics)
    dataloader_statistics = get_statistics(test_data_loader)
    print("test data loader statistics:", dataloader_statistics)
    # average_val_deviation = evaluator.get_average_deviation(valid_data_loader)
    # print("average training deviation:", average_trn_deviation)
    # print("average validation deviation:", average_val_deviation)
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


