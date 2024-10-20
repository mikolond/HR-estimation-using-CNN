import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self,path, filenames_vid, filenames_txt, n, fs_list):
        self.path = path
        self.filenames_vid = filenames_vid
        self.filenames_txt = filenames_txt
        self.fs_list = fs_list
        self.n = n
        self.current_file = 0
        self.current_shot = 0 # start of the sequence of the shots in the video
        self.current_opened_video = None
        self.current_opened_txt = None
    
    def __len__(self):
        return len(self.filenames_vid)

    def __getitem__(self, idx):
        
        return 