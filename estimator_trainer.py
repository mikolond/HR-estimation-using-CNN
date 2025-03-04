import torch
from model_estimator import Estimator
from estimator_loss import EstimatorLoss
from estimator_dataset_loader import EstimatorDatasetLoader
import numpy as np
import os
import time

DEBUG = False


class EstimatorTrainer:
    def __init__(self, train_data_loader, valid_data_loader, device, lr = 0.01, batch_size=1, num_epochs = 5):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.model = Estimator()
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = EstimatorLoss().to(device)

    def create_batch(self, batch_size=1):
        sequence = np.zeros((batch_size,150))
        hr_data = np.zeros((batch_size))
        epoch_done = False
        for i in range(batch_size):
            data = self.train_data_loader.get_sequence()
            sequence[i] = data[0]
            hr_data[i] = data[1] / 60
            epoch_done = not self.train_data_loader.next_sequence()
        return sequence, hr_data, epoch_done

    def validate(self):
        self.model.eval()
        valid_loss = 0
        valid_count = 0
        epoch_done = False
        with torch.no_grad():
            while not epoch_done:
                sequence, hr_data, epoch_done = self.create_batch(1)
                sequence = sequence.reshape(1,150,1).transpose(0,2,1)
                x = torch.tensor(sequence).float().to(self.device)
                output = self.model(x)
                loss = self.criterion(output, hr_data)
                valid_count += 1
                valid_loss += loss.item()
        self.model.train()
        print(f"Validation Loss: {valid_loss/valid_count}")
        


    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_count = 0
            epoch_done = False
            epoch_start = time.time()
            while not epoch_done:
                sequence, hr_data, epoch_done = self.create_batch(self.batch_size)
                sequence = sequence.reshape(self.batch_size,150,1).transpose(0,2,1)
                self.optimizer.zero_grad()
                x = torch.tensor(sequence).float().to(self.device)
                output = self.model(x)
                loss = self.criterion(output, hr_data)
                loss.backward()
                self.optimizer.step()
                train_count += 1
                train_loss += loss.item()
            after_train = time.time()
            print(f"Epoch: {epoch}, Loss: {train_loss/train_count}")
            self.validate()
            after_valid = time.time()
            self.train_data_loader.reset()
            self.valid_data_loader.reset()
            if DEBUG:
                print("Train time:", after_train - epoch_start)
                print("Valid time:", after_valid - after_train)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    import csv
    import yaml
    dataset_path = os.path.join("datasets", "estimator_synthetic")
    folders_path = os.path.join(dataset_path, "data.csv")
    folders = []
    with open(folders_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            folders.append(row)

    benchmark_path = os.path.join("benchmarks", "benchmark_synthetic.yaml")
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

    # create training data loader
    train_data_loader = EstimatorDatasetLoader(dataset_path, train_videos_list, N=150, step_size=1)
    
    # create validation data loader
    valid_data_loader = EstimatorDatasetLoader(dataset_path, valid_videos_list, N=150, step_size=1)

    device = input("Device to train on: ")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + device)
    # device = torch.device("cpu")

    trainer = EstimatorTrainer(train_data_loader, valid_data_loader, device, batch_size=1, num_epochs=20, lr=0.0001)
    trainer.train()
    weights_path = os.path.join("output","estimator_weights","weights_exp1.pth")
    trainer.save_model(weights_path)

