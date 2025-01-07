import torch
from model import Extractor
import numpy as np
from dataset_loader import DatasetLoader
import matplotlib.pyplot as plt

N = 200 # length of the frame sequence

class ExtractorInference:
    def __init__(self, model_path, data_loader, device):
        self.model_path = model_path
        self.data_loader = data_loader
        self.device = device
        self.model = Extractor().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def infer(self):
        frames = self.data_loader.get_sequence()
        x = torch.tensor(frames.transpose(0, 3, 1, 2), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(x).reshape(N)
        output_numpy = output.detach().cpu().numpy()
        print("output shape", output.shape)
        self.plot_output(output_numpy)

    def plot_output(self, output_numpy):
        plt.figure()
        plt.plot(output_numpy)
        plt.title("Output")
        plt.show()

if __name__ == "__main__":
    loader = DatasetLoader("C:\\projects\\dataset_creator_test_output", ["video_0"], N=N)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    inference = ExtractorInference("model.pth", loader, device)
    inference.infer()


