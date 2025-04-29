import torch
from Models.estimator_model import Estimator
# from Models.estimator_model_spetlik import Estimator
from Loss.estimator_loss import EstimatorLoss
from Datasets_handlers.Estimator.dataset_loader import EstimatorDatasetLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import csv

DEBUG = False


class EstimatorTrainer:
    def __init__(self, train_data_loader, valid_data_loader, device, lr = 0.01, batch_size=1, num_epochs = 5, best_model_path = None, output_path = None):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.best_model_path = best_model_path
        if not os.path.exists(self.best_model_path):
            os.makedirs(self.best_model_path)
        self.best_loss = float("inf")

        self.model = Estimator()
        # self.model.setup()
        self.model.to(device)
        # init optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),amsgrad=False, lr=self.lr, weight_decay=0)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.01)
        # init loss function
        # self.criterion = EstimatorLoss().to(device)
        # self.criterion = PearsonLoss().to(device)
        # self.criterion = torch.nn.MSELoss().to(device)
        self.criterion = torch.nn.L1Loss().to(device)
        self.output_path = output_path

        self.lr_decay = False
        self.decay_epochs = []
        self.decay_rate = 0.1
    
    def set_lr_decay(self, decay_epochs, decay_rate):
        self.lr_decay = True
        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
    
    def set_patience(self, patience):
        self.patience = patience
        self.epochs_without_improvement = 0
        self.early_stopping = False
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def create_batch(self, data_loader, batch_size=1):
        sequence = np.zeros((batch_size,self.train_data_loader.N))
        hr_data = np.zeros((batch_size))
        epoch_done = False
        j = 0
        for i in range(batch_size):
            seq, hr = data_loader.get_sequence()
            sequence[i] = seq
            hr_data[i] = hr/60
            epoch_done = not data_loader.next_sequence()
            progress = data_loader.get_progress()
            j += 1
            if epoch_done:
                break
            # print(f"Progress: {progress[0]}/{progress[1]}", end="\r")
        return sequence[:j], hr_data[:j], epoch_done, progress
    

    def validate(self):
        self.model.eval()
        valid_loss = 0
        valid_count = 0
        epoch_done = False
        with torch.no_grad():
            while not epoch_done:
                sequence, hr_data, epoch_done,_ = self.create_batch(self.valid_data_loader,self.batch_size)
                current_batch_size = len(sequence)
                sequence = sequence.reshape(current_batch_size,self.valid_data_loader.N,1).transpose(0,2,1)
                x = torch.tensor(sequence).float().to(self.device)
                output = self.model(x).reshape(current_batch_size)
                loss = self.criterion(output, torch.tensor(hr_data).to(self.device))
                # print("valid predicted hr:", int(output.item()*60), "real hr:", int(hr_data[0]*60),"loss:",loss.item()*60)
                valid_count += 1
                valid_loss += loss.item()
        self.model.train()
        valid_loss = valid_loss/valid_count * 60
        torch.save(self.model.state_dict(), os.path.join(self.best_model_path, "last_estimator_weights.pth"))
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            torch.save(self.model.state_dict(), os.path.join(self.best_model_path, "best_estimator_weights.pth"))
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.early_stopping = True
                print("Early stopping")
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
                sequence, hr_data, epoch_done, progress = self.create_batch(self.train_data_loader, self.batch_size)
                current_batch_size = len(sequence)
                sequence = sequence.reshape(current_batch_size,self.train_data_loader.N,1).transpose(0,2,1)

                x = torch.tensor(sequence).float().to(self.device)
                # print("shape of x:",x.shape)
                output = self.model(x).reshape(current_batch_size)
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
            train_loss = train_loss / train_count * 60
            print(f"Epoch: {epoch}, Loss: {train_loss}")
            valid_loss = self.validate()
            train_loss_log.append(train_loss)
            valid_loss_log.append(valid_loss)
            after_valid = time.time()
            self.train_data_loader.reset()
            self.valid_data_loader.reset()
            if DEBUG:
                print("Train time:", after_train - epoch_start)
                print("Valid time:", after_valid - after_train)
            if self.early_stopping:
                print("Early stopping")
                break
            if self.lr_decay and epoch in self.decay_epochs:
                self.lr *= self.decay_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.decay_rate
                print(f"Learning rate decayed to {self.lr}")
        print("Training finished, best loss:", self.best_loss)
        # plot training loss
        plt.plot(train_loss_log)
        plt.xlabel("Epoch")
        plt.ylabel("Loss [bpm]")
        plt.title("Training loss - Estimator")
        plt.savefig(os.path.join(self.output_path, "estimator_train_loss.png"))
        plt.clf()
        # plot validation loss
        plt.plot(valid_loss_log)
        plt.xlabel("Epoch")
        plt.ylabel("Loss [bpm]")
        plt.title("Validation loss - Estimator")
        plt.savefig(os.path.join(self.output_path, "estimator_valid_loss.png"))
        plt.close()

        # save the losses to csv file
        with open(os.path.join(self.output_path, "estimator_train_log.csv"), mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['epoch', 'train_loss', 'valid_loss'])
            for i in range(len(train_loss_log)):
                csv_writer.writerow([i, train_loss_log[i], valid_loss_log[i]])


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
if __name__ == "__main__":
    model = Estimator()
    # model.setup()
    print("Number of parameters in the model: ", count_params(model))
    # import csv
    # import yaml
    # dataset_path = os.path.join("datasets", "estimator_pure_pure_extractor_150")
    # folders_path = os.path.join(dataset_path, "data.csv")
    # folders = []
    # with open(folders_path, 'r') as file:
    #     reader = csv.reader(file)
    #     for row in reader:
    #         folders.append(row)

    # benchmark_path = os.path.join("benchmarks", "benchmark_pure.yaml")
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

    # # create training data loader
    # train_data_loader = EstimatorDatasetLoader(dataset_path, train_videos_list, N=300, step_size=50)
    
    # # create validation data loader
    # valid_data_loader = EstimatorDatasetLoader(dataset_path, valid_videos_list, N=300, step_size=300)

    # device = input("Device to train on: ")
    # if not torch.cuda.is_available():
    #     device = torch.device("cpu")
    # else:
    #     device = torch.device("cuda:" + device)
    # # device = torch.device("cpu")
    # output_path = os.path.join("output", "estimator_pure_median")
    # trainer = EstimatorTrainer(train_data_loader, valid_data_loader, device, batch_size=600, num_epochs=1000, lr=0.01, best_model_path=output_path, output_path=output_path)
    # trainer.set_patience(300)
    # trainer.set_lr_decay([100,200,300,400,500,600,700], 0.7)
    # # trainer.load_model(os.path.join("output","estimator_pure_weights", "best_model.pth"))
    # trainer.train()
    # # weights_path = os.path.join("output","estimator_weights","weights_latest.pth")
    # # trainer.save_model(weights_path)

