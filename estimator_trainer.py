import torch
from model_estimator import Estimator
from estimator_loss import EstimatorLoss
from estimator_dataset_loader import EstimatorDatasetLoader
import numpy as np
import os
import time

DEBUG = False


class EstimatorTrainer:
    def __init__(self, train_data_loader, valid_data_loader, device, lr = 0.01, batch_size=1, num_epochs = 5, best_model_path = None):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.best_model_path = best_model_path
        self.best_loss = float("inf")

        self.model = Estimator()
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = EstimatorLoss().to(device)
        # self.criterion = torch.nn.MSELoss().to(device)

    def create_batch(self, batch_size=1):
        sequence = np.zeros((batch_size,self.train_data_loader.N))
        hr_data = np.zeros((batch_size))
        epoch_done = False
        for i in range(batch_size):
            seq, hr = self.train_data_loader.get_sequence()
            sequence[i] = seq
            hr_data[i] = hr/60
            epoch_done = not self.train_data_loader.next_sequence()
            progress = self.train_data_loader.get_progress()
            # print(f"Progress: {progress[0]}/{progress[1]}", end="\r")
        return sequence, hr_data, epoch_done, progress

    def validate(self):
        self.model.eval()
        valid_loss = 0
        valid_count = 0
        epoch_done = False
        with torch.no_grad():
            while not epoch_done:
                sequence, hr_data, epoch_done,_ = self.create_batch(1)
                sequence = sequence.reshape(1,self.valid_data_loader.N,1).transpose(0,2,1)
                x = torch.tensor(sequence).float().to(self.device)
                output = self.model(x).reshape(1)
                loss = self.criterion(output, torch.tensor(hr_data).to(self.device))
                # print("valid predicted hr:", int(output.item()*60), "real hr:", int(hr_data[0]*60),"loss:",loss.item()*60)
                valid_count += 1
                valid_loss += loss.item()
        self.model.train()
        valid_loss = valid_loss/valid_count * 60
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            torch.save(self.model.state_dict(), os.path.join(self.best_model_path, "best_model.pth"))
        print(f"Validation Loss: {valid_loss}")
        


    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_count = 0
            epoch_done = False
            epoch_start = time.time()
            while not epoch_done:
                sequence, hr_data, epoch_done, progress = self.create_batch(self.batch_size)
                sequence = sequence.reshape(self.batch_size,self.train_data_loader.N,1).transpose(0,2,1)

                x = torch.tensor(sequence).float().to(self.device)
                output = self.model(x).reshape(self.batch_size)
                loss = self.criterion(output, torch.tensor(hr_data, dtype=torch.float).to(self.device))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_count += 1
                train_loss += loss.item()
                print(f"Epoch progress: {progress[0]}/{progress[1]}", end="\r")
                # print(f"predicted hr:{(output.detach().cpu().numpy()*60).astype(np.int32)}real hr:{(np.array(hr_data)*60).astype(np.int32),"loss:",(loss.detach().cpu().numpy()*60).astype(np.int32)}")
                # print("train predicted hr:", int(output.item()*60), "real hr:", int(hr_data[0]*60),"loss:",loss.item()*60)
            after_train = time.time()
            print(f"Epoch: {epoch}, Loss: {train_loss/train_count*60}")
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
    dataset_path = os.path.join("datasets", "estimator_ecg_fitness_latest")
    folders_path = os.path.join(dataset_path, "data.csv")
    folders = []
    with open(folders_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            folders.append(row)

    benchmark_path = os.path.join("benchmarks", "benchmark_ecg.yaml")
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
    train_data_loader = EstimatorDatasetLoader(dataset_path, train_videos_list, N=300, step_size=50)
    
    # create validation data loader
    valid_data_loader = EstimatorDatasetLoader(dataset_path, valid_videos_list, N=300, step_size=300)

    device = input("Device to train on: ")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + device)
    # device = torch.device("cpu")

    trainer = EstimatorTrainer(train_data_loader, valid_data_loader, device, batch_size=100, num_epochs=50, lr=0.01, best_model_path="output/estimator_weights")
    trainer.train()
    weights_path = os.path.join("output","estimator_weights","weights_ecg2.pth")
    trainer.save_model(weights_path)

