import torch
from model_estimator import Estimator
from estimator_dataset_loader import EstimatorDatasetLoader
import numpy as np
import os
import matplotlib.pyplot as plt

class EstimatorEval:
    def __init__(self, weights_path, device):
        self.model = Estimator().to(device)
        self.model.eval()
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.device = device


    def infer(self, sequence):
        sequence = sequence.reshape(1,150,1).transpose(0,2,1)
        x = torch.tensor(sequence).float().to(self.device)
        output = self.model(x)
        return output.item()


if __name__ == "__main__":
    weights_path = os.path.join("output","estimator_weights","weights_exp1.pth")
    device = torch.device("cuda:0")
    dataset_path = os.path.join("datasets", "estimator_synthetic")
    train_videos_list = ["video_120.csv"]

    data_loader = EstimatorDatasetLoader(dataset_path, train_videos_list, N=150, step_size=50)

    estimator = EstimatorEval(weights_path,device)

    for i in range(20):
        sequence, real_hr = data_loader.get_sequence()
        # print("sequence:",sequence)
        predicted_hr = estimator.infer(sequence)
        fig1 = plt.figure()
        plt.plot(sequence)
        plt.title("Sequence")
        # save fig1
        fig1.savefig(os.path.join("trash","sequence.png"))
        
        print(f"predicted hr:{predicted_hr}, real hr:{real_hr/60}")
        data_loader.next_sequence()


