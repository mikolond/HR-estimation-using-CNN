import torch
from model import Extractor
import numpy as np

from dataset_loader import DatasetLoader
import matplotlib.pyplot as plt


N = 88 # length of the frame sequence

if __name__ == "__main__":

    loader = DatasetLoader("C:\\projects\\dataset_synthetic_output", ["video_0"], N = N)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device)
    model = Extractor().to(device)

    model.load_state_dict(torch.load("model.pth"))

    # load 1 video
    frames = loader.get_sequence()

    x = torch.tensor(frames.transpose(0,3,1,2), dtype=torch.float32).to(device)
    

    with torch.no_grad():
        output = model(x).reshape(N)

    output_numpy = output.detach().cpu().numpy()  # Move to CPU for plotting

    print("output shape", output.shape)

    # plot the output
    plt.figure()
    plt.plot(output_numpy)
    plt.title("Output")
    # show plot
    plt.show()


