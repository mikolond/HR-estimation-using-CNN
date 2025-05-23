import torch
from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import csv
from utils import load_model_class
import argparse
plot_counter = 0

# CONFIG_PATH = os.path.join("config_files", "statistics", "config_eval_pure1.yaml")
# CONFIG_PATH = os.path.join("config_files", "model5", "config_eval_pure_ecg.yaml")
CONFIG_PATH = os.path.join("config_files", "cross_val", "pure", "new", "config_eval_split1.yaml")

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
    
    # get rif of all values not from inerval 40-240
    statistics["count"] = len(hr_data)
    # get all outliers
    outliers = hr_data[(hr_data < 40) | (hr_data > 240)]
    statistics["outliers"] = outliers
    hr_data = hr_data[(hr_data >= 40) & (hr_data <= 240)]
    outliers_count = statistics["count"] - len(hr_data)
    print("outliers count", outliers_count)

    hr_data = np.array(hr_data)
    average = np.mean(hr_data)
    std_deviation = np.std(hr_data)
    mean_deviation = np.mean(np.abs(hr_data - average))
    statistics["mean_deviation"] = mean_deviation
    statistics["average"] = average
    statistics["median"] = np.median(hr_data)
    statistics["median deviation"] = np.mean(np.abs(hr_data - statistics["median"]))
    statistics["std_deviation"] = std_deviation
    statistics["min"] = np.min(hr_data)
    statistics["max"] = np.max(hr_data)

    return statistics

class EstimatorEval:
    def __init__(self, extractor_model, extractor_weights_path, estimator_weights_path, device, N, output_path, estimator_model_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        Estimator = load_model_class(estimator_model_path, "Estimator")
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
    

    
    def evaluate(self, data_loader, tag = "unknown", save_predicitons=False):
        loss = {}
        # print("evaluating dataset")
        loss["rmse"], loss["mae"], loss["pearson"] = self.validate(data_loader, tag, save_predicitons)
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
                # prediction = get_max_freq_padded(extractor_output, 30, hr_data, 0, pad_factor=10) * 60
                epoch_done = not data_loader.next_sequence()
                ground_truth.append(hr_data)
                predicted.append(prediction)
                progress = data_loader.get_progress()
                print(f"Progress: {progress[0]}/{progress[1]}", end="\r")
                # get_max_freq_padded(extractor_output, 30, hr_data, prediction, pad_factor=10)

        return ground_truth, predicted
    
    def validate(self, data_loader, tag, save_predicitons):
        ground_truth, predicted = self.make_predictions(data_loader)
        if save_predicitons:
            with open(os.path.join(self.output_path, "predictions_" + tag + ".txt"), 'w') as f:
                f.write("ground_truth predicted\n")
                for i in range(len(predicted)):
                    f.write(f"{ground_truth[i]} {predicted[i]}\n")
        
        plot_pearson(ground_truth, predicted, os.path.join(self.output_path), tag)
        min_hr = 40
        max__hr_data = max(max(ground_truth), max(predicted))
        print("max__hr_data", max__hr_data)
        max_hr = max(160, max__hr_data+10)//10*10
        print("max_hr", max_hr)
        errors_per_class = get_per_class_error(ground_truth, predicted,min_hr=min_hr, max_hr=max_hr)
        plot_per_class_error(errors_per_class, os.path.join(self.output_path), tag, min_hr=min_hr, max_hr=max_hr)
        
        errors = np.array(predicted) - np.array(ground_truth)
        # computes the rmse
        rmse = np.sqrt(np.mean(errors**2))
        # computes the mae
        mae = np.mean(np.abs(errors))
        # computes the pearsion correlation
        pearson = np.corrcoef(ground_truth, predicted)[0, 1]
        return rmse, mae, pearson
    
def get_per_class_error(ground_truth, predicted, min_hr = 40, max_hr=240, step_hr=10):
    '''Calculate the error per class (e.g., per 10 bpm) and return the average error.'''
    errors = np.array(predicted) - np.array(ground_truth)
    # print("errors", errors)
    # print("ground_truth", ground_truth)
    # print("predicted", predicted)
    classes = np.arange(min_hr, max_hr, step_hr)
    errors_per_class = []
    for i in range(len(classes)-1):
        class_errors = errors[(ground_truth >= classes[i]) & (ground_truth < classes[i+1])]
        if len(class_errors) > 0:
            errors_per_class.append(np.mean(np.abs(class_errors)))
        else:
            errors_per_class.append(0)
    return errors_per_class

def plot_per_class_error(errors_per_class, save_path, tag, min_hr = 40, max_hr=240, step_hr=10):
    '''Plot the error per class.'''
    classes_lower_bounds = np.arange(min_hr, max_hr, step_hr)
    class_labels = [f'{lower}-{lower + step_hr}' for lower in classes_lower_bounds[:-1]]

    plt.figure()
    plt.bar(classes_lower_bounds[:-1], errors_per_class, width=8)
    plt.xlabel("Heart Rate Range [bpm]")
    plt.ylabel("Error [bpm]")
    plt.title("Error per Class")
    plt.xticks(classes_lower_bounds[:-1], class_labels, rotation=45, ha='right')
    plt.grid()
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    file_name = os.path.join(save_path, "error_per_class_" + tag + ".png")
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
        
    
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

    return max_freq

def plot_pearson(ground_truth, predicted, save_path, tag):
    """Plots ground truth vs predicted with larger font sizes."""
    plt.figure()
    pearson = np.corrcoef(ground_truth, predicted)[0, 1]
    minimum = min(min(ground_truth), min(predicted))
    maximum = max(max(ground_truth), max(predicted))
    # Round axis limits to nearest 5
    step = 10
    minimum = int(minimum - 5)
    maximum = int(maximum + 5)
    plt.xlim(minimum, maximum)
    plt.ylim(minimum, maximum)
    plt.plot([minimum, maximum], [minimum, maximum], 'r--', label='Ideal Prediction') # Added label
    plt.xticks(np.arange(minimum, maximum, step), rotation=45, fontsize=14) # Increased font size
    plt.yticks(np.arange(minimum, maximum + 5, step), fontsize=14) # Increased font size
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(ground_truth, predicted, label='Predictions') # Added label to scatter
    plt.title(f"{tag}-set, Pearson correlation R={pearson:.2f}", fontsize=16) # Increased font size
    plt.xlabel("Ground Truth [bpm]", fontsize=14) # Increased font size
    plt.ylabel("Predicted [bpm]", fontsize=14) # Increased font size
    plt.legend(fontsize=12) # Added legend with increased font size
    namefile = "pearson_correlation_" + tag + ".png"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, namefile), bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_sequence(sequence,freqs,fft, real_hr,predicted, save_path):
    global plot_counter
    plt.figure(figsize=(8,3))
    plt.plot(sequence)
    plt.title("Sequence")
    plt.xlabel("Frame number")
    plt.ylabel("Amplitude")
    plt.tight_layout()
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
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("config_path", type=str, help="Path to the config file", default=None)
    args = parser.parse_args()
    if args.config_path is None:
        raise Exception("No config path provided")
    else:
        if not os.path.exists(args.config_path):
            raise Exception("Config path does not exist")
        else:
            config_path = args.config_path

    config_data = yaml.safe_load(open(config_path, "r"))
    data = config_data["data"]
    weights = config_data["weights"]
    models = config_data["models"]
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


    device = input("Device to use: ")
    if not torch.cuda.is_available() or device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + device)
    save_predictions_to_txt = config_data["save_predictions_to_txt"]

    extractor_model_path = models["extractor_model_path"]
    estimator_model_path = models["estimator_model_path"]
    Extractor = load_model_class(extractor_model_path, "Extractor")
    extractor_model = Extractor()
    extractor_model.load_state_dict(torch.load(extractor_weights_path, map_location=device))

    evaluator = EstimatorEval(extractor_model, extractor_weights_path,estimator_weights_path, device, seq_length, output_path, estimator_model_path)
    print("evaluating test data")
    loss_tst = evaluator.evaluate(test_data_loader, tag = "test", save_predicitons=save_predictions_to_txt)
    print("test loss:", loss_tst)
    
    print("evaluating train data")
    loss_tr = evaluator.evaluate(train_data_loader, tag = "train", save_predicitons=save_predictions_to_txt)
    print("train loss:", loss_tr)
    print("evaluating validation data")
    loss_val = evaluator.evaluate(valid_data_loader, tag = "validation", save_predicitons=save_predictions_to_txt)
    print("validation loss:", loss_val)

    train_data_loader.reset()
    valid_data_loader.reset()
    test_data_loader.reset()

    # open the result.csv in the output path
    result_path = os.path.join(output_path, "result.csv")
    if not os.path.exists(result_path):
        with open(result_path, 'w') as file:
            file.write("train_rmse, train_mae, train_pearson, valid_rmse, valid_mae, valid_pearson, test_rmse, test_mae, test_pearson\n")
    with open(result_path, 'a') as file:
        file.write(f"{loss_tr['rmse']}, {loss_tr['mae']}, {loss_tr['pearson']}, {loss_val['rmse']}, {loss_val['mae']}, {loss_val['pearson']}, {loss_tst['rmse']}, {loss_tst['mae']}, {loss_tst['pearson']}\n")
    print("Results saved to", result_path)
    print("Evaluation done")

    # print("Evaluating data loaders statistics")
    # dataloader_statistics = get_statistics(train_data_loader)
    # print("train data loader statistics:", dataloader_statistics)
    # dataloader_statistics = get_statistics(valid_data_loader)
    # print("validation data loader statistics:", dataloader_statistics)
    # dataloader_statistics = get_statistics(test_data_loader)
    # print("test data loader statistics:", dataloader_statistics)