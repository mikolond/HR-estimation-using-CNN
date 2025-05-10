import yaml
import csv
import os
import numpy as np
import torch

from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
from Trainers.extractor_trainer import ExtractorTrainer

CONFIG_PATH = os.path.join("config_files", "final_experiments", "config_extractor_ecg_new_model.yaml")
# CONFIG_PATH = os.path.join("config_files", "final_experiments", "config_extractor_pure_new_model.yaml")
# CONFIG_PATH = os.path.join("config_files", "final_experiments", "config_extractor_ecg_original_model.yaml")
# CONFIG_PATH = os.path.join("config_files", "final_experiments", "config_extractor_pure_original_model.yaml")




def train_extractor(config_path):
    # Load the YAML file
    config_data = yaml.safe_load(open(config_path, "r"))
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
    model_path = config_data["extractor_model_path"]
    if not os.path.exists(model_path):
        raise Exception("Model path does not exist")

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
    trainer = ExtractorTrainer(train_data_loader, valid_data_loader, device, model_path, output_path=output_path, learning_rate=lr, batch_size=batch_size, 
                               num_epochs=num_epochs, patience = patience, N=train_sequence_length, hr_data=hr_data, cum_batch_size=cum_batch_size, 
                               lr_decay=lr_decay, decay_rate=decrease_lr, decay_epochs=lr_decay_epochs, weights_path = output_path)
    if config_data["load_model"]:
        if os.path.exists(config_data["load_model_path"]):
            trainer.load_model(config_data["load_model_path"])
    
    # old_model_path = os.path.join("output","pure_exp22", "best_extractor_weights.pth")
    # trainer.transfer_weights(old_model_path)
    # lr1 = 1e-4
    # lr2 = 1e-6
    # layers = [("bn_input", lr1),("conv1",lr1),("conv2",lr1),("conv3",lr1),("conv4",lr1),("conv5",lr2),("conv6",lr2),("conv7",lr2),("conv_last",lr2)]
    # trainer.make_custom_optimizer(layers)
    print("Training started")
    print("Model:", config_data["extractor_model_path"])
    print("Output path:", output_path)
    print("optimizer stats:", optimizer)
    print("train stats:", train)
    print("valid stats:", valid)

    trainer.train()

if __name__ == "__main__":
    train_extractor(CONFIG_PATH)