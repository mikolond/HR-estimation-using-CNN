from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
import numpy as np
import os

if __name__ == "__main__":
    import yaml
    import csv
    config_data = yaml.safe_load(open("config_files/config_ecg_local.yaml"))
    data = config_data["data"]
    optimizer = config_data["optimizer"]
    hr_data = config_data["hr_data"]
    train = config_data["train"]
    valid = config_data["valid"]
    # load data
    weights_path = data["weights_dir"]
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
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

    # train_data_loader.next_sequence()
    hr_list = train_data_loader.get_hr_list()
    print("hr_list:",hr_list)

    hist = np.histogram(hr_list, bins=range(40, 200, 1), density=True)
    print("hist:",hist)