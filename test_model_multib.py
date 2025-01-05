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
        ground_truth.append(float(line)/60)
    return ground_truth



# load 1 video

video_path = "test_videos/120_30_60.avi"
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

# load 2 video

video2_path = "test_videos/80_30_60.avi"
cap = cv2.VideoCapture(video2_path)

# get video parameters
N_vid2 = load_N_frames(cap,192, 128, N)
print("lenght of sequence",len(N_vid2))

# load ground_truth
f = open(video2_path.replace(".avi", ".txt"), "r")
N_truth2 = load_N_ground_truth(f, N)
print("lenght of ground truth",len(N_truth2))
print("ground truth",N_truth2)


# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device",device)

model = Extractor().to(device)
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

# example of 2 batch input

# load model 
N_vid_combined = N_vid + N_vid2
x = torch.tensor(np.array(N_vid_combined).transpose(0, 3, 2, 1)).float().to(device)  # shape (2N, C, H, W)
print("shape of x",x.shape)
f_true = torch.tensor(np.array([N_truth[0],N_truth2[0]])).float().to(device)  # shape (N)
fs = f_s  # shape (1)
delta = 5/60
f_range = np.array([1, 150]) / 60
sampling_f = 1/60

print("f_true:",f_true)
# load loss
criterion = ExtractorLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

i = 0
train_count = 100
for i in range(train_count):
    optimizer.zero_grad()  # Clear gradients
    output = model(x).reshape(2,N)


    print("output shape", output.shape)

    output_numpy = output.detach().cpu().numpy()  # Move to CPU for plotting
    # create figure with output plot
    output_numpy = output_numpy.reshape(2*N)
    plt.figure()
    plt.plot(output_numpy)
    plt.title("Output" + str(i))
    # save figure
    plt.savefig("graphs/output" + str(i) + ".png")

    loss = criterion(output, f_true, fs, delta, sampling_f, f_range)
    print("loss ", i, ":", loss)
    loss.backward()
    optimizer.step()

    # multiply learning rate by 0.99 every iteration
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.8 

pass

# output_numpy = output.detach().numpy()
# # create figure with output plot
# plt.figure()
# plt.plot(output_numpy[0])
# plt.title("Output")
# # save figure
# plt.savefig("output2.png")



# loss = criterion(output, f_true, fs, delta, sampling_f, f_range)

# print("loss2:",loss)
# loss.backward()
# optimizer.step()

# output = model(x,N).reshape(1,N)
# print("output shape",output.shape)

# output_numpy = output.detach().numpy()
# # create figure with output plot
# plt.figure()
# plt.plot(output_numpy[0])
# plt.title("Output")
# # save figure
# plt.savefig("output3.png")

# loss = criterion(output, f_true, fs, delta, sampling_f, f_range)
# print("loss3:",loss)

# loss.backward()
# optimizer.step()

# print("loss",loss)
