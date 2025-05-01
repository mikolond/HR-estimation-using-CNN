import yaml
import csv
import os
import numpy as np
import torch

from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
from extractor_trainer import ExtractorTrainer

# CONFIG_PATH = os.path.join("config_files", "synthetic", "config_extractor_synthetic.yaml")
# CONFIG_PATH = os.path.join("config_files", "pure", "config_extractor_pure_halmos_exp24.yaml")
# CONFIG_PATH = os.path.join("config_files", "pure_local", "config_extractor_exp22_better.yaml")
CONFIG_PATH = os.path.join("config_files", "latent_model_test", "config_extractor_pure_halmos_latent3_exp5.yaml")


if __name__ == "__main__":
    # Load the YAML file
    config_data = yaml.safe_load(open(CONFIG_PATH, "r"))
    data = config_data["data"]
    optimizer = config_data["optimizer"]
    hr_data = config_data["hr_data"]
    train = config_data["train"]
    valid = config_data["valid"]
    # load data
    output_path = data["output_dir"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
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
    train_sequence_length = train["sequence_length"]
    train_shift = train["shift"]
    train_augment = train["augment"]
    train_data_loader = DatasetLoader(dataset_path, train_videos_list, N=train_sequence_length, step_size=train_shift, augmentation=train_augment)
    
    # create validation data loader
    valid_sequence_length = valid["sequence_length"]
    valid_shift = valid["shift"]
    valid_augment = valid["augment"]
    valid_data_loader = DatasetLoader(dataset_path, valid_videos_list, N=valid_sequence_length, step_size=valid_shift, augmentation=valid_augment)

    # load HR data setting for loss function
    delta = hr_data["delta"]/60
    f_range = np.array(hr_data["frequency_range"])/60
    sampling_f = hr_data["sampling_frequency"]/60
    hr_data = {"delta": delta, "f_range": f_range, "sampling_f": sampling_f}


    # load data for training
    lr = float(optimizer["lr"])
    batch_size = optimizer["batch_size"]
    cum_batch_size = optimizer["cumulative_batch_size"]
    num_epochs = optimizer["max_epochs"]
    patience = optimizer["patience"]
    decrease_lr = optimizer["decrease_lr"]
    lr_decay = optimizer["lr_decay"]
    lr_decay_epochs = optimizer["lr_decay_epochs"]
    device = input("Device to train on: ")
    if not torch.cuda.is_available() or device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + device)
    trainer = ExtractorTrainer(train_data_loader, valid_data_loader, device, output_path=output_path, learning_rate=lr, batch_size=batch_size, 
                               num_epochs=num_epochs, patience = patience, N=train_sequence_length, hr_data=hr_data, cum_batch_size=cum_batch_size, 
                               lr_decay=lr_decay, decay_rate=decrease_lr, decay_epochs=lr_decay_epochs, weights_path = output_path)
    if config_data["load_model"]:
        if os.path.exists(config_data["load_model_path"]):
            trainer.load_model(config_data["load_model_path"])
    trainer.train()