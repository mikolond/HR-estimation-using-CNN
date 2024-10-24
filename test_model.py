from model import Extractor
from loss import ExtractorLoss
from matplotlib import pyplot as plt

import math
import torch
import cv2
import numpy as np
N = 50 # length of the frame sequence

def load_N_frames(cap, width, height, N):
    '''
    Load N frames from the video and reshape the resolution
    params: cap : cv2.VideoCapture object
            width : int
            height : int
            N : int
    return: list of N frames
    '''
    frames = []
    for i in range(N):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    return frames

def load_N_ground_truth(f, N):
    '''
    Load N ground truth values from the file
    params: f : file object
            N : int
    return: list of N ground truth values
    '''
    ground_truth = []
    for i in range(N):
        line = f.readline()
        if not line:
            break
        ground_truth.append(float(line))
    return ground_truth



# load 1 video

video_path = "test_videos/0.8_20_10.avi"
cap = cv2.VideoCapture(video_path)

# get video parameters
frames_c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
f_s = int(cap.get(cv2.CAP_PROP_FPS))

N_vid = load_N_frames(cap,192, 128, N)
print("lenght of sequence",len(N_vid))
print("shape of sequence",N_vid[0].shape)

# load ground_truth
f = open(video_path.replace(".avi", ".txt"), "r")
N_truth = load_N_ground_truth(f, N)
print("lenght of ground truth",len(N_truth))
print("ground truth",N_truth)

# load model

model = Extractor()
model.init_weights()

# generate sinusoidal signal from the fs and f_true to plot ground true
t = np.arange(0, N/f_s, 1/f_s)
f_true = N_truth[0]
fs = f_s
c = 2 * np.pi * f_true  / fs
amplitude = 50
signal = [100 + math.sin(i*c)*amplitude for i in range(N)]
plt.figure()
plt.plot(signal)
plt.title("Ground True")
plt.savefig("ground_true.png")


# try feedforward of the model

x = torch.tensor(np.array(N_vid).transpose(0,3,2,1)).float()  # shape (N, C, H, W)
print("shape of x",x.shape)
f_true = N_truth # shape (N)
fs = f_s  # shape (1)
delta = 5/60
f_range = np.array([1, 150]) / 60
sampling_f = 1/60




output = model(x,N).reshape(1,N)
print("output shape",output.shape)

output_numpy = output.detach().numpy()
# create figure with output plot
plt.figure()
plt.plot(output_numpy[0])
plt.title("Output")
# save figure
plt.savefig("output1.png")



# load loss
criterion = ExtractorLoss()

loss = criterion(output, f_true, fs, delta, sampling_f, f_range)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print("loss1:",loss)
loss.backward()
optimizer.step()

output = model(x,N).reshape(1,N)
print("output shape",output.shape)

output_numpy = output.detach().numpy()
# create figure with output plot
plt.figure()
plt.plot(output_numpy[0])
plt.title("Output")
# save figure
plt.savefig("output2.png")



loss = criterion(output, f_true, fs, delta, sampling_f, f_range)

print("loss2:",loss)
loss.backward()
optimizer.step()

output = model(x,N).reshape(1,N)
print("output shape",output.shape)

output_numpy = output.detach().numpy()
# create figure with output plot
plt.figure()
plt.plot(output_numpy[0])
plt.title("Output")
# save figure
plt.savefig("output3.png")

loss = criterion(output, f_true, fs, delta, sampling_f, f_range)
print("loss3:",loss)

loss.backward()
optimizer.step()

print("loss",loss)
