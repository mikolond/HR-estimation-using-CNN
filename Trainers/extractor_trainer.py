import torch
from Loss.extractor_loss import ExtractorLoss
from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import csv
from utils import load_model_class
from utils import transfer_weights


DEBUG = False

class ExtractorTrainer:
    def __init__(self, train_data_loader, valid_data_loader, device, model_path, output_path=None, learning_rate=0.0001, batch_size=1, num_epochs=100, patience = 30, debug=False, N=100, hr_data=None, cum_batch_size=1, lr_decay = False, decay_rate = 0.5, decay_epochs = [1], weights_path = None):
        if hr_data is not None:
            self.hr_data = hr_data
        else:
            raise ValueError("hr_data is not provided")
        self.lr_decay = lr_decay
        self.decay_rate = decay_rate
        self.decay_epochs = decay_epochs
        self.cum_batch_size = cum_batch_size
        self.training_sequence_length = N
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.debug = debug
        Extractor = load_model_class(model_path, "Extractor")
        self.model = Extractor().to(self.device)
        self.loss_fc = ExtractorLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.7)
        self.validation_loss_log = []
        self.current_epoch = 0
        self.current_epoch_time = 0
        self.last_epoch_time = 0

        self.weights_path = weights_path
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        self.patience = patience
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.valid_loss_log_path = os.path.join(output_path,"valid_loss_log.csv")

        if train_data_loader.get_progress()[1] == 0:
            raise ValueError("No data in the training dataset")
        if train_data_loader.get_progress()[1] < self.cum_batch_size:
            raise ValueError("Cummulatve batch size is larger than the training dataset")
        
    def transfer_weights(self, model_path):
        if not os.path.exists(model_path):
            raise ValueError("Model path does not exist")
        transfer_weights(model_path, self.model, self.device)
        self.model.to(self.device)
    
    def make_custom_optimizer(self, layer_names_lrs: list[tuple[str, float]]):
        params = []
        assigned_param_names = set()

        for layer_name, lr in layer_names_lrs:
            # Collect parameters for the specified layer
            layer_params = []
            for name, param in self.model.named_parameters():
                if name.startswith(layer_name + "."):  # Match layer and its sub-layers
                    layer_params.append(param)
                    assigned_param_names.add(name)  # Keep track of assigned params

            if layer_params:
                params.append({"params": layer_params, "lr": lr})

        remaining_params = []
        for name, param in self.model.named_parameters():
            if name not in assigned_param_names:
                if self.learning_rate is not None:
                    remaining_params.append(param)
                else:
                    raise ValueError(
                        f"Parameter '{name}' has no learning rate specified and "
                        "no default learning rate provided."
                    )
        if remaining_params:
            params.append({"params": remaining_params, "lr": self.learning_rate})

        self.optimizer = torch.optim.Adam(params, weight_decay=1e-4)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def get_delta_from_hr_list(self, hr_list):
        min_hr = min(hr_list)
        max_hr = max(hr_list)
        median_hr = np.median(hr_list)
        delta = max(abs(min_hr - median_hr), abs(max_hr - median_hr)) / 60 # convert bpm to Hz
        delta = min(delta, self.hr_data["delta"])
        delta = max(delta, self.hr_data["delta"]/2)
        return delta

    def train(self):
        # path_to_save = os.path.join(self.weights_path, "model_epoch_-1.pth")
        # torch.save(self.model.state_dict(), path_to_save)
        #  create another folder for model weights
        best_valid_loss = float("inf")
        epochs_without_improvement = 0
        train_loss_log = []
        for i in range(self.num_epochs):
            train_loss = 0
            self.current_epoch = i
            start_time = time.time()
            self.model.train()
            epoch_done = False
            train_counter = 1
            while not epoch_done:
                epoch_start = time.time()
                sequence, f_true, fs, n_of_sequences, epoch_done, deltas = self.create_batch(self.train_data_loader)
                if n_of_sequences != 0:
                    before_x = time.time()
                    x = torch.tensor(sequence.reshape(n_of_sequences * self.training_sequence_length, 192, 128, 3).transpose(0, 3, 1, 2)).float().to(self.device)
                    if self.debug:
                        print("shape of x", x.shape)
                    f_true = torch.tensor(f_true).float().to(self.device)
                    before_infer = time.time()
                    output = self.model(x)
                    output = output.reshape(n_of_sequences, self.training_sequence_length)
                    if self.debug:
                        print("output shape", output.shape)
                    before_loss = time.time()
                    f_range = self.hr_data["f_range"]
                    sampling_f = self.hr_data["sampling_f"]
                    loss = self.loss_fc(output, f_true, fs, deltas, sampling_f, f_range)
                    train_loss += loss.item()
                    self.log_progress(loss.item(), start_time)
                    loss = loss / self.cum_batch_size
                    before_backward = time.time()
                    loss.backward()
                    if train_counter % self.cum_batch_size == 0 or epoch_done:
                        before_optimizer = time.time()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

  
                    # print all times
                    if self.debug:
                        print("Time for loading batch:", before_x - epoch_start)
                        print("Time taken for x:", before_infer - before_x)
                        print("Time taken for inference:", before_loss - before_infer)
                        print("Time taken for loss calculation:", before_backward - before_loss)
                        print("Time taken for backward pass:", before_optimizer - before_backward)
                        print("Time taken for optimizer step:", time.time() - before_optimizer)
                train_counter += 1
            train_loss /= train_counter
            train_loss_log.append(train_loss)
            valid_loss = self.validate()
            log_file = open(self.valid_loss_log_path,"a")
            log_file.write(str(valid_loss.item()) + "\n")
            log_file.close()
            torch.save(self.model.state_dict(), os.path.join(self.weights_path, "last_extractor_weights.pth"))
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, "best_extractor_weights.pth"))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement > self.patience:
                    print("Early stopping")
                    break
            self.train_data_loader.reset()
            self.valid_data_loader.reset()
            self.last_epoch_time = self.current_epoch_time
            print("\nepoch", i, "done,","validation loss:", self.validation_loss_log[-1], ",train loss:", train_loss)
            # save weights of this epoch
            # path_to_save = os.path.join(self.weights_path, "model_epoch_" + str(i) + ".pth")
            # torch.save(self.model.state_dict(), path_to_save)
            # decrease learning rate
            if self.lr_decay and  i in self.decay_epochs:
                self.learning_rate *= self.decay_rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
        print("training done, best loss:", best_valid_loss)
        # plot training loss
        plt.plot(train_loss_log)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss - Extractor")
        plt.savefig(os.path.join(self.output_path, "extractor_train_loss.png"))
        plt.clf()
        # plot validation loss
        plt.plot(self.validation_loss_log)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation loss - Extractor")
        plt.savefig(os.path.join(self.output_path, "extractor_valid_loss.png"))
        plt.close()

        with open(os.path.join(self.output_path, "extractor_train_log.csv"), mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['epoch', 'train_loss', 'valid_loss'])
            for i in range(len(train_loss_log)):
                csv_writer.writerow([i, train_loss_log[i], self.validation_loss_log[i]])



    def create_batch(self, data_loader=None):
        sequence_length = data_loader.N
        sequence = np.zeros((self.batch_size, sequence_length, 192, 128, 3))
        f_true = np.zeros((self.batch_size))
        fs = np.zeros((self.batch_size))
        deltas = np.zeros((self.batch_size))
        n_of_sequences = 0
        next_seq_out = None
        epoch_done = False
        for j in range(self.batch_size):
            cur_seq = data_loader.get_sequence()
            sequence[j] = cur_seq
            hr_list = data_loader.get_hr_list()
            deltas[j] = self.get_delta_from_hr_list(hr_list)
            f_true[j] = data_loader.get_hr() / 60
            fs[j] = data_loader.get_fps()
            next_seq_out = data_loader.next_sequence()
            n_of_sequences = j + 1
            if next_seq_out is None and j < self.batch_size:
                if self.debug:
                    print("epoch done, but batch not full")
                break
        if next_seq_out is None:
            epoch_done = True
        return sequence[:n_of_sequences], f_true[:n_of_sequences], fs[:n_of_sequences], n_of_sequences, epoch_done, deltas

    def log_progress(self, loss, start_time):
        epoch_progress = self.train_data_loader.get_progress()
        time_passed = time.time() - start_time
        self.current_epoch_time = time_passed/epoch_progress[0] * epoch_progress[1]
        if self.current_epoch == 0:
            self.last_epoch_time = self.current_epoch_time
        estimated_time = time_passed / epoch_progress[0] * (epoch_progress[1] - epoch_progress[0]) + self.last_epoch_time * (self.num_epochs - self.current_epoch - 1)
        estimated_time_minutes = estimated_time // 60
        estimated_time_hours = estimated_time_minutes // 60
        percentage_progress = epoch_progress[0] / epoch_progress[1] * 100
        alpha = 0.99  # Smoothing factor
        if not hasattr(self, 'ema_loss'):
            self.ema_loss = loss
        else:
            self.ema_loss = alpha * self.ema_loss + (1 - alpha) * loss
        print("EMA loss:{:.4f}".format(self.ema_loss), ",epoch progress:", int(percentage_progress), "% ,eta:", estimated_time_hours, "h and", estimated_time_minutes % 60, "m", end="\r")

    def validate(self):
        print("\nvalidation")
        self.model.eval()
        with torch.no_grad():
            self.valid_data_loader.reset()
            sequence_length = self.valid_data_loader.N
            valid_loss = 0
            valid_count = 0
            validation_done = False
            while not validation_done:
                sequence, f_true, fs, n_of_sequences, validation_done, deltas = self.create_batch(self.valid_data_loader)
                if n_of_sequences == 0:
                    break
                x = torch.tensor(sequence.reshape(n_of_sequences * sequence_length, 192, 128, 3).transpose(0, 3, 1, 2)).float().to(self.device)
                if self.debug:
                    print("shape of x", x.shape)
                f_true = torch.tensor(f_true).float().to(self.device)
                output = self.model(x).reshape(n_of_sequences, sequence_length)
                f_range = self.hr_data["f_range"]
                sampling_f = self.hr_data["sampling_f"]
                valid_loss += self.loss_fc(output, f_true, fs, deltas, sampling_f, f_range)
                progress = self.valid_data_loader.get_progress()
                percentage_progress = progress[0] / progress[1] * 100
                valid_count += 1
                print("loss:{:.4f}".format(valid_loss.item()/valid_count), ",progress:", int(percentage_progress),"%" , end="\r")
                validation_done = not self.valid_data_loader.next_sequence()
            valid_loss /= valid_count
            self.validation_loss_log.append(valid_loss.detach().cpu().numpy().item())
        return valid_loss

    def save_model(self):
        user = input("Save model? y/n")
        if user == "y":
            torch.save(self.model.state_dict(), "model.pth")
