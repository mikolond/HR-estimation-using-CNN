import torch
from model import Extractor
from loss import ExtractorLoss
from dataset_loader import DatasetLoader
import numpy as np
import time
import matplotlib.pyplot as plt

N = 70 # length of the frame sequence
delta = 5/60 # offset from the true frequency
f_range = np.array([20, 220]) / 60 # all possible frequencies
sampling_f = 1/60 # sampling frequency in loss calculating

LEARING_RATE = 0.00001

DEBUG = False

class ExtractorTrainer:
    def __init__(self, train_data_loader, valid_data_loader, device, learning_rate=0.0001, batch_size=2, num_epochs=5, debug=False):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.debug = debug
        self.model = Extractor().to(self.device)
        self.model.init_weights()
        self.loss_fc = ExtractorLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.validation_loss_log = []
        self.current_epoch = 0
        self.current_epoch_time = 0
        self.last_epoch_time = 0

    def train(self):
        for i in range(self.num_epochs):
            self.current_epoch = i
            start_time = time.time()
            self.model.train()
            epoch_done = False
            while not epoch_done:
                self.optimizer.zero_grad()
                sequence, f_true, fs, n_of_sequences, epoch_done = self.create_batch()
                if n_of_sequences != 0:
                    x = torch.tensor(sequence.reshape(n_of_sequences * N, 192, 128, 3).transpose(0, 3, 1, 2)).float().to(self.device)
                    if self.debug:
                        print("shape of x", x.shape)
                    f_true = torch.tensor(f_true).float().to(self.device)
                    output = self.model(x).reshape(n_of_sequences, N)
                    if self.debug:
                        print("output shape", output.shape)
                    loss = self.loss_fc(output, f_true, fs, delta, sampling_f, f_range)
                    self.log_progress(loss, start_time)
                    loss.backward()
                    self.optimizer.step()
            self.validate()
            self.train_data_loader.reset()
            self.valid_data_loader.reset()
            self.last_epoch_time = self.current_epoch_time
            print("epoch", i, "done")
            print("validation loss", self.validation_loss_log)
        self.plot_validation_loss()

    def create_batch(self):
        sequence = np.zeros((self.batch_size, N, 192, 128, 3))
        f_true = np.zeros((self.batch_size))
        fs = np.zeros((self.batch_size))
        n_of_sequences = 0
        epoch_done = False
        for j in range(self.batch_size):
            cur_seq = self.train_data_loader.get_sequence()
            sequence[j] = cur_seq
            f_true[j] = self.train_data_loader.get_hr() / 60
            fs[j] = self.train_data_loader.get_fps()
            epoch_done = not self.train_data_loader.next_sequence()
            n_of_sequences = j + 1
            if epoch_done and j < self.batch_size:
                if self.debug:
                    print("epoch done, but batch not full")
                break
        return sequence[:n_of_sequences], f_true[:n_of_sequences], fs[:n_of_sequences], n_of_sequences, epoch_done

    def log_progress(self, loss, start_time):
        epoch_progress = self.train_data_loader.progress()
        time_passed = time.time() - start_time
        self.current_epoch_time = time_passed/epoch_progress[0] * epoch_progress[1]
        if self.current_epoch == 0:
            self.last_epoch_time = self.current_epoch_time
        estimated_time = time_passed / epoch_progress[0] * (epoch_progress[1] - epoch_progress[0]) + self.last_epoch_time * (self.num_epochs - self.current_epoch - 1)
        estimated_time_minutes = estimated_time // 60
        estimated_time_hours = estimated_time_minutes // 60
        percentage_progress = epoch_progress[0] / epoch_progress[1] * 100
        print("loss:{:.4f}".format(loss.item()), ",progress:", int(percentage_progress), "% ,eta:", estimated_time_hours, "h and", estimated_time_minutes % 60, "m")

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            self.valid_data_loader.reset()
            valid_loss = 0
            valid_count = 0
            validation_done = False
            while not validation_done:
                sequence = self.valid_data_loader.get_sequence()
                f_true = [self.valid_data_loader.get_hr() / 60]
                fs = [self.valid_data_loader.get_fps()]
                x = torch.tensor(sequence.transpose(0, 3, 1, 2)).float().to(self.device)
                if self.debug:
                    print("shape of x", x.shape)
                f_true = torch.tensor(f_true).float().to(self.device)
                output = self.model(x).reshape(1, N)
                valid_loss += self.loss_fc(output, f_true, fs, delta, sampling_f, f_range)
                valid_count += 1
                validation_done = not self.valid_data_loader.next_sequence()
            valid_loss /= valid_count
            print("valid loss", valid_loss)
            self.validation_loss_log.append(valid_loss.detach().cpu().numpy().item())

    def plot_validation_loss(self):
        plt.figure()
        plt.plot(self.validation_loss_log)
        plt.title("Validation loss")
        plt.show()

    def save_model(self):
        user = input("Save model? y/n")
        if user == "y":
            torch.save(self.model.state_dict(), "model.pth")

if __name__ == "__main__":
    train_videos_list = []
    for i in range(0,75):
        train_videos_list.append("video_" + str(i))
    valid_videos_list = []
    for i in range(75, 84):
        valid_videos_list.append("video_" + str(i))
    train_data_loader = DatasetLoader("C:\\projects\\dataset_creator_test_output", train_videos_list, N=N, step_size=40)
    valid_data_loader = DatasetLoader("C:\\projects\\dataset_creator_test_output", valid_videos_list, N=N, step_size=40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    trainer = ExtractorTrainer(train_data_loader, valid_data_loader, device, debug=DEBUG)
    trainer.train()
    trainer.save_model()
    