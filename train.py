import torch
from model import Extractor
from loss import ExtractorLoss
from dataset_loader import DatasetLoader
import numpy as np

N = 88 # length of the frame sequence
delta = 5/60 # offset from the true frequency
f_range = np.array([20, 220]) / 60 # all possible frequencies
sampling_f = 1/60 # sampling frequency in loss calculating

DEBUG = True


if __name__ == "__main__":


    train_data_loader = DatasetLoader("C:\\projects\\dataset_synthetic_output", ["video_0", "video_1","video_2"], N = N)
    valid_data_loader = DatasetLoader("C:\\projects\\dataset_synthetic_output", ["video_3"], N = N)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device)
    model = Extractor().to(device)
    model.init_weights()
    loss_fc = ExtractorLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    batch_size = 3
    num_epochs = 100

    epoch_done = False
    for i in range(num_epochs):
        while not epoch_done:
            optimizer.zero_grad()
            # create batch
            sequence = np.zeros((batch_size, N, 192, 128, 3))
            f_true = np.zeros((batch_size))
            fs = np.zeros((batch_size))
            n_of_sequences = 0
            print("sequence shape",sequence.shape)

            for j in range(batch_size):
                cur_seq = train_data_loader.get_sequence()
                print("cur_seq shape",cur_seq.shape,"j = ",j, "sequence shape",sequence[j].shape)
                sequence[j] = cur_seq
                f_true[j] = train_data_loader.get_hr()/60 # convert to Hz
                fs[j] = train_data_loader.get_fps()
                epoch_done = not train_data_loader.next_sequence()
                n_of_sequences = j + 1
                if epoch_done and j < batch_size:
                    print("epoch done, but batch not full")
                    break
            # create incomplete batch
            if n_of_sequences != 0:
                sequence = sequence[:n_of_sequences]
                f_true = f_true[:n_of_sequences]
                fs = fs[:n_of_sequences]

            print("sequence shape",sequence.shape)
            
            x = torch.tensor(sequence.reshape(n_of_sequences * N, 192, 128, 3).transpose(0, 3, 1, 2)).float().to(device)  # shape (batch_size *N, C, H, W)
            print("shape of x",x.shape)
            f_true = torch.tensor(f_true).float().to(device)

            # forward pass
            output = model(x).reshape(n_of_sequences, N)
            print("output shape", output.shape)


            loss = loss_fc(output, f_true, fs, delta, sampling_f, f_range)
            print("loss",loss)
            # backward pass
            loss.backward()
            # optimize
            optimizer.step()

        # validation
        with torch.no_grad():
            valid_data_loader.reset()
            valid_loss = 0
            valid_count = 0
            validation_done = False
            while not validation_done:
                sequence = valid_data_loader.get_sequence()
                f_true = [valid_data_loader.get_hr()/60] # convert to Hz
                fs = [valid_data_loader.get_fps()]
                x = torch.tensor(sequence.transpose(0, 3, 1, 2)).float().to(device)
                print("shape of x",x.shape)
                f_true = torch.tensor(f_true).float().to(device)
                output = model(x).reshape(1, N)
                print("f_true",f_true)
                valid_loss += loss_fc(output, f_true, fs, delta, sampling_f, f_range)
                valid_count += 1
                validation_done = not valid_data_loader.next_sequence()

            valid_loss /= valid_count
            print("valid loss",valid_loss)
        epoch_done = False

    pass



    
