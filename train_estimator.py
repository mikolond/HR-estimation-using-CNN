import yaml
import csv
import os
import numpy as np
import torch
torch.manual_seed(0)

from Datasets_handlers.Estimator.dataset_creator import DatasetCreator
from Datasets_handlers.Estimator.dataset_loader import EstimatorDatasetLoader

from estimator_trainer import EstimatorTrainer

# CONFIG_PATH = os.path.join("config_files", "synthetic", "config_estimator_synthetic.yaml")
CONFIG_PATH = os.path.join("config_files", "pure_local", "config_estimator_exp24.yaml")



# CONFIG_PATH = os.path.join("config_files", "estimator_augment.yaml")


if __name__ == "__main__":
    # Load the YAML file
    config_data = yaml.safe_load(open(CONFIG_PATH, "r"))
    data = config_data["data"]

    device = input("Device to train on: ")
    if not torch.cuda.is_available() or device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + device)

    estimator_dataset_path = data["estimator_dataset_dir"]
    create_new_dataset = config_data["create_new_dataset"]
    if not os.path.exists(estimator_dataset_path) or create_new_dataset:
        # if estimator dataet does not exist, create it
        extractor_dataset_path = data["extractor_dataset_dir"]
        if not os.path.exists(extractor_dataset_path):
            raise Exception("Extractor and Estimator datasets does not exist")
        extractor_weights_path = data["extractor_weights"]
        dataset_creator_N = config_data["dataset_creator_N"]
        augmentation = config_data["dataset_creator_augmentation"]
        # create dataset_creator
        dataset_creator = DatasetCreator(extractor_weights_path, device, extractor_dataset_path, estimator_dataset_path, dataset_creator_N, augmentation)
        # create dataset
        dataset_creator.create_dataset()
    else:
        print("Estimator dataset already exists")
    # load data
    dataset_path = estimator_dataset_path
    folders_path = os.path.join(dataset_path, "data.csv")
    folders = []
    with open(folders_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            folders.append(row)
        
    benchmark_path = data["benchmark"]
    benchmark = yaml.safe_load(open(benchmark_path))
    train_folders = benchmark["trn"]
    valid_folders = benchmark["val"]
    train_videos_list = np.array([])
    valid_videos_list = np.array([])

    for idx in train_folders:
        train_videos_list = np.append(train_videos_list, np.array(folders[idx]))
    
    for idx in valid_folders:
        valid_videos_list = np.append(valid_videos_list, np.array(folders[idx]))

    # add .csv after every video name
    for i in range(len(train_videos_list)):
        train_videos_list[i] += ".csv"
    for i in range(len(valid_videos_list)):
        valid_videos_list[i] += ".csv"

    train_n = config_data["train"]["sequence_length"]
    train_shift = config_data["train"]["shift"]

    valid_n = config_data["valid"]["sequence_length"]
    valid_shift = config_data["valid"]["shift"]

    # create training data loader
    train_data_loader = EstimatorDatasetLoader(dataset_path, train_videos_list, N=train_n, step_size=train_shift)
    
    # create validation data loader
    valid_data_loader = EstimatorDatasetLoader(dataset_path, valid_videos_list, N=valid_n, step_size=valid_shift)

    output_path = data["output_dir"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    optimizer = config_data["optimizer"]
    batch_size = optimizer["batch_size"]
    lr = float(optimizer["lr"])
    max_epochs = optimizer["max_epochs"]
    patience = optimizer["patience"]
    decrease_lr = optimizer["decrease_lr"]
    lr_decay = optimizer["lr_decay"]
    lr_decay_epochs = optimizer["lr_decay_epochs"]


    trainer = EstimatorTrainer(train_data_loader, valid_data_loader, device, batch_size=batch_size, num_epochs=max_epochs, lr=lr, best_model_path=output_path, output_path=output_path)
    load_model = config_data["load_model"]
    if load_model:
        model_path = config_data["load_model_path"]
        trainer.load_model(model_path)

    if decrease_lr:
        trainer.set_lr_decay(lr_decay_epochs, lr_decay)
    trainer.set_patience(patience)

    trainer.train()

