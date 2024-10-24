import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


class MyDataset(Dataset):
    def __init__(self,path, filenames_vid, filenames_txt, n, batch_size, flag):
        self.path = path
        self.filenames_vid = filenames_vid
        self.filenames_txt = filenames_txt
        self.current_file = 0
        self.current_shot = 0 # start of the sequence of the shots in the video
        self.current_opened_video = None
        self.current_opened_txt = None
        self.flag = flag
        self.n = n
        self.batch_size = batch_size
        self.n_s = 0
        if flag == "debug":
            # load all videos and get the lenghts of the sequences from all videos
            for i in range(len(self.filenames_vid)):
                cap = cv2.VideoCapture(self.path + self.filenames_vid[i])
                frames_c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.n_s += frames_c - n +1 
                cap.release()
    
    def __len__(self):
        return self.n_s

    def __getitem__(self, idx):
        if self.flag == "debug":
            # load 1 video
            if self.current_opened_video is None:
                self.current_opened_video = cv2.VideoCapture(self.path + self.filenames_vid[self.current_file])
                self.current_opened_txt = open(self.path + self.filenames_txt[self.current_file], "r")
            # get video parameters
            frames_c = int(self.current_opened_video.get(cv2.CAP_PROP_FRAME_COUNT))
            f_s = int(self.current_opened_video.get(cv2.CAP_PROP_FPS))
            # load N frames
            frames = self.load_N_frames(self.current_opened_video, 192, 128, self.n, self.current_shot)
            # load ground_truth
            ground_truth = self.load_N_ground_truth(self.current_opened_txt, self.n, self.current_shot)
            # update the current shot
            self.current_shot += 1
            if self.current_shot > frames_c - self.n:
                self.current_opened_video.release()
                self.current_opened_txt.close()
                self.current_opened_video = None
                self.current_opened_txt = None
                self.current_file += 1
                self.current_shot = 0
            return frames, ground_truth
        else:
            # load 1 video
            cap = cv2.VideoCapture(self.path + self.filenames_vid[self.current_file])
            # get video parameters
            frames_c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            f_s = int(cap.get(cv2.CAP_PROP_FPS)
            # load N frames
            frames = self.load_N_frames(cap, 192, 128, self.n, self.current_shot)
            # load ground_truth
            f = open(self.path + self.filenames_txt[self.current_file], "r")
            ground_truth = self.load_N_ground_truth(f, self.n, self.current_shot)
            # update the current shot
            self.current_shot += 1
            if self.current_shot > frames_c - self.n:
                cap.release()
                f.close()
                self.current_file += 1
                self.current_shot = 0
            return frames, ground_truth
        return 