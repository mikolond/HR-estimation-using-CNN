import torch
from Models.estimator_model import Estimator
from Models.extractor_model import Extractor
from Loss.estimator_loss import EstimatorLoss
# from Datasets_handlers.Estimator.dataset_loader import EstimatorDatasetLoader
from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time

DEBUG = False


class EstimatorTrainer:
    def __init__(self, train_data_loader, valid_data_loader, device, extractor_model, patience=None, output_path = None, lr = 0.01, batch_size=1, num_epochs = 5, best_model_path = None):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.best_model_path = best_model_path
        self.best_loss = float("inf")

        self.model = Estimator()
        self.extractor_model = extractor_model
        self.extractor_model.to(device)
        self.extractor_model.eval()
        # for param in self.extractor_model.parameters():
        #     param.requires_grad = False
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = EstimatorLoss().to(device)
        self.sequence_length = train_data_loader.N
        # self.criterion = torch.nn.MSELoss().to(device)
        self.output_path = output_path
        self.patience = patience
        self.epochs_without_improvement = 0
        self.early_stopping = False

    def infer_extractor(self, sequence):
        with torch.no_grad():
            x = torch.tensor(sequence.reshape(self.N, 192, 128, 3).transpose(0, 3, 1, 2)).float().to(self.device)
            output = self.extractor_model(x).reshape(self.N).cpu().numpy()
        return output

    def create_batch(self, data_loader, batch_size=1):
        batch_output = np.zeros((batch_size, self.sequence_length))
        f_true_out = np.zeros((batch_size))
        n_of_sequences = 0
        next_seq_out = None

        for i in range(batch_size):
            sequence = data_loader.get_sequence()
            f_true = data_loader.get_hr() / 60
            next_seq_out = data_loader.next_sequence()
            progress = data_loader.get_progress()
            
            with torch.no_grad():
                x = torch.tensor(sequence.reshape(self.sequence_length, 192, 128, 3).transpose(0, 3, 1, 2)).float().to(self.device)
                # print("shape of x", x.shape)
                output = self.extractor_model(x).reshape(self.sequence_length).cpu().numpy()
            batch_output[i] = output
            f_true_out[i] = f_true
            n_of_sequences = i + 1
        if next_seq_out is None:
            epoch_done = True
        else:
            epoch_done = False
            # print(f"Batch crating : {i}/{self.batch_size}", end="\r")

        return batch_output[:n_of_sequences], f_true_out[:n_of_sequences], epoch_done, progress
    

    def validate(self):
        self.model.eval()
        valid_loss = 0
        valid_count = 0
        epoch_done = False
        with torch.no_grad():
            while not epoch_done:
                x, f_true, epoch_done, progress = self.create_batch(self.valid_data_loader, self.batch_size)
                seq_length = x.shape[0]
                x = torch.tensor(x.reshape(seq_length,self.train_data_loader.N,1).transpose(0,2,1), dtype=torch.float).to(self.device)
                output = self.model(x).reshape(seq_length)
                loss = self.criterion(output, torch.tensor(f_true, dtype=torch.float).to(self.device))
                valid_count += 1
                valid_loss += loss.item()
        self.model.train()
        valid_loss = valid_loss / valid_count * 60
        if valid_loss < self.best_loss:
            self.epochs_without_improvement = 0
            self.best_loss = valid_loss
            torch.save(self.model.state_dict(), os.path.join(self.best_model_path, "best_estimator_weights.pth"))
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {self.num_epochs}")
                self.early_stopping = True
        print(f"Validation Loss: {valid_loss}")
        return valid_loss
        


    def train(self):
        train_loss_log = []
        valid_loss_log = []
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            train_count = 0
            epoch_done = False
            epoch_start = time.time()
            while not epoch_done:
                x, f_true, epoch_done, progress = self.create_batch(self.train_data_loader, self.batch_size)
                seq_length = x.shape[0]
                x = torch.tensor(x.reshape(seq_length, self.train_data_loader.N,1).transpose(0,2,1), dtype=torch.float).to(self.device)
                # print("f_true shape", f_true.shape)
                # print("progress shape", progress)
                # print("x shape", x.shape)
                output = self.model(x).reshape(seq_length)
                # print("output shape", output.shape)
                # print("output:", output, "f_true:", f_true)
                loss = self.criterion(output, torch.tensor(f_true, dtype=torch.float).to(self.device))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_count += 1
                train_loss += loss.item()
                print(f"Epoch progress: {progress[0]}/{progress[1]}", end="\r")
            after_train = time.time()
            print(f"Epoch: {epoch}, Loss: {train_loss / train_count * 60}")
            train_loss_log.append(train_loss / train_count * 60)
            valid_loss = self.validate()
            valid_loss_log.append(valid_loss)
            after_valid = time.time()
            self.train_data_loader.reset()
            self.valid_data_loader.reset()
            if self.early_stopping:
                print("Early stopping")
                break
            if DEBUG:
                print("Train time:", after_train - epoch_start)
                print("Valid time:", after_valid - after_train)
        # plot training loss
        plt.plot(train_loss_log)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss - Estimator")
        plt.savefig(os.path.join(self.output_path, "estimator_train_loss.png"))
        plt.clf()
        # plot validation loss
        plt.plot(valid_loss_log)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation loss - Estimator")
        plt.savefig(os.path.join(self.output_path, "estimator_valid_loss.png"))
        plt.close()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    import yaml
    import csv
    config_data = yaml.safe_load(open("config_files/config_pure_halmos_decreasing.yaml"))
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

    trainer = EstimatorTrainer(train_data_loader, valid_data_loader, device, extractor_model, patience=patience, output_path=output_path, batch_size=batch_size, num_epochs=num_epochs, lr=lr, best_model_path=weights_path)
    trainer.train()
    weights_path = os.path.join("output","estimator_weights","weights_latest.pth")
    trainer.save_model(weights_path)

