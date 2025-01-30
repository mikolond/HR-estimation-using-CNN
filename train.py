import torch
from model import Extractor
from loss import ExtractorLoss
from dataset_loader import DatasetLoader
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter

N =  160# length of the frame sequence
delta = 5/60 # offset from the true frequency
f_range = np.array([35, 240]) / 60 # possible frequencies boundaries
sampling_f = 1/60 # sampling frequency in loss calculating
BATCH_SIZE = 1
ARTIFICIAL_BATCH_SIZE = 1
LEARING_RATE = 1e-4
NUM_EPOCHS = 10

DECRESING_LEARNING_RATE = True
DEECREASING_RATE = 0.5
NUM_EPOCHS_TO_DECRESING = 1

DEBUG = False

class ExtractorTrainer:
    def __init__(self, train_data_loader, valid_data_loader, device, learning_rate=0.0001, batch_size=1, num_epochs=5, debug=False, N=100):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.debug = debug
        self.model = Extractor().to(self.device)
        self.loss_fc = ExtractorLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, foreach=False)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.7)
        self.validation_loss_log = []
        self.current_epoch = 0
        self.current_epoch_time = 0
        self.last_epoch_time = 0
        if not os.path.exists(os.path.join('net-'+str(learning_rate)[0:10])):
            os.makedirs(os.path.join('net-'+str(learning_rate)[0:10]))

        self.writer = SummaryWriter(os.path.join('net-'+str(learning_rate)[0:10]+'/'))
        self.train_log_counter = 0


    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def train(self):
        print("training parameters:", "batch size:", self.batch_size, "learning rate:", self.learning_rate, "num epochs:", self.num_epochs, "artificial batch size:", ARTIFICIAL_BATCH_SIZE,"decreasing learning rate:", DECRESING_LEARNING_RATE, "decreasing rate:", DEECREASING_RATE, "num epochs to decreasing:", NUM_EPOCHS_TO_DECRESING, "N:", N, "delta:", delta, "f_range:", f_range, "sampling_f:", sampling_f)
        #  create another folder for model weights
        if not os.path.exists("model_weights"):
            os.makedirs("model_weights")
        for i in range(self.num_epochs):
            self.current_epoch = i
            start_time = time.time()
            self.model.train()
            epoch_done = False
            train_counter = 1
            while not epoch_done:
                epoch_start = time.time()
                sequence, f_true, fs, n_of_sequences, epoch_done = self.create_batch()
                if n_of_sequences != 0:
                    before_x = time.time()
                    x = torch.tensor(sequence.reshape(n_of_sequences * N, 192, 128, 3).transpose(0, 3, 1, 2)).float().to(self.device)
                    if self.debug:
                        print("shape of x", x.shape)
                    f_true = torch.tensor(f_true).float().to(self.device)
                    before_infer = time.time()
                    output = self.model(x).reshape(n_of_sequences, N)
                    if self.debug:
                        print("output shape", output.shape)
                    before_loss = time.time()
                    loss = self.loss_fc(output, f_true, fs, delta, sampling_f, f_range)
                    self.log_progress(loss.item(), start_time)
                    self.writer.add_scalar("Loss/train", loss, self.train_log_counter)
                    self.train_log_counter += 1
                    before_backward = time.time()
                    loss.backward()
                    if train_counter % ARTIFICIAL_BATCH_SIZE == 0:
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param.grad.data /= ARTIFICIAL_BATCH_SIZE
                        before_optimizer = time.time()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

  
                    # print all times
                    if self.debug:
                        print("Time taken for x:", before_infer - before_x)
                        print("Time taken for inference:", before_loss - before_infer)
                        print("Time taken for loss calculation:", before_backward - before_loss)
                        print("Time taken for backward pass:", before_optimizer - before_backward)
                        print("Time taken for optimizer step:", time.time() - before_optimizer)
                train_counter += 1
            self.validate()
            self.train_data_loader.reset()
            self.valid_data_loader.reset()
            self.last_epoch_time = self.current_epoch_time
            print("\nepoch", i, "done")
            print("validation loss", self.validation_loss_log)
            # save weights of this epoch
            torch.save(self.model.state_dict(), "model_weights/model_epoch_" + str(i) + ".pth")
            # decrease learning rate
            if DECRESING_LEARNING_RATE and i % NUM_EPOCHS_TO_DECRESING == 0:
                self.learning_rate *= DEECREASING_RATE
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
        self.writer.flush()
        # self.plot_validation_loss()

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
                progress = self.valid_data_loader.progress()
                percentage_progress = progress[0] / progress[1] * 100
                valid_count += 1
                print("loss:{:.4f}".format(valid_loss.item()/valid_count), ",progress:", int(percentage_progress),"%" , end="\r")
                validation_done = not self.valid_data_loader.next_sequence()
            valid_loss /= valid_count
            self.writer.add_scalar("Loss/valid", valid_loss, self.current_epoch)
            self.validation_loss_log.append(valid_loss.detach().cpu().numpy().item())

    def save_model(self):
        user = input("Save model? y/n")
        if user == "y":
            torch.save(self.model.state_dict(), "model.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("N", type=int, help="Length of the frame sequence", default=N)
    parser.add_argument("batch_size", type=int, help="Batch size", default=BATCH_SIZE)
    parser.add_argument("learning_rate", type=float, help="Learning rate", default = LEARING_RATE)
    parser.add_argument("num_epochs", type=int, help="Number of epochs", default=NUM_EPOCHS)
    parser.add_argument("train_path", type=str, help="Path to the train dataset", default="dataset/train")
    parser.add_argument("valid_path", type=str, help="Path to the validation dataset", default="dataset/valid")
    parser.add_argument("--device", type=str, help="Device to train on", default="cuda:0")
    args = parser.parse_args()
    train_path = args.train_path
    valid_path = args.valid_path
    train_videos_list = os.listdir(train_path)
    valid_videos_list = os.listdir(valid_path)
    train_data_loader = DatasetLoader(train_path, train_videos_list, N=args.N, step_size=args.N, augmentation=True)
    valid_data_loader = DatasetLoader(valid_path, valid_videos_list, N=args.N, step_size=args.N)
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    trainer = ExtractorTrainer(train_data_loader, valid_data_loader, device,learning_rate=args.learning_rate, debug=DEBUG, batch_size=args.batch_size, num_epochs=args.num_epochs)
    trainer.train()
