import os
import cv2
import random
from copy import deepcopy
import numpy as np



class DatasetLoader:
    def __init__(self, dataset_path, videos, N = 100):
        self.dataset_path = dataset_path
        self.videos = videos
        self.N = N

        # check if the dataset path exists
        if not os.path.exists(self.dataset_path):
            raise Exception("Dataset path does not exist")
        
        # check if the videos are in the dataset path
        for video in self.videos:
            if not os.path.exists(self.dataset_path + "\\" + video):
                raise Exception("Video path does not exist")
            
        # calculate number of N dequences in the dataset
        self.N_sequences = 0
        for video in self.videos:
            images_count = len(os.listdir(self.dataset_path + "\\" + video)) -2
            self.N_sequences += images_count - self.N + 1
        
        # shuffle the videos
        self.current_video_idx = 0
        random.shuffle(self.videos)
        self.current_video = self.videos[self.current_video_idx]
        self.current_image = 0
        self.current_video_frames_count = len(os.listdir(self.dataset_path + "\\" + self.current_video)) - 2
        if self.N > self.current_video_frames_count:
            raise Exception("N is greater than the number of frames in the video")
        
        self.frames = np.zeros((N, *cv2.imread(self.dataset_path + "\\" + self.current_video + "\\" + "0.png").shape), dtype=np.uint8)
        # load first N frames
        for i in range(N):
            frame = cv2.imread(self.dataset_path + "\\" + self.current_video + "\\" + str(i) + ".png")
            self.frames[i] = frame

        self.current_N_sequence = 0

    def load_next_frame(self):
        '''
        Load the next frame
        return: frame
        '''
        self.current_N_sequence += 1
        frame_to_load = self.current_image + self.N
        if frame_to_load < self.current_video_frames_count:
            frame = cv2.imread(self.dataset_path + "\\" + self.current_video + "\\" + str(frame_to_load) + ".png")
            self.frames = np.roll(self.frames, -1, axis=0)
            self.frames[-1] = frame
            self.current_image += 1
            return True
        else:
            # load the next video
            self.current_video_idx += 1
            if self.current_video_idx >= len(self.videos):
                return False
            self.current_video = self.videos[self.current_video_idx]
            self.current_video_frames_count = len(os.listdir(self.dataset_path + "\\" + self.current_video)) - 2
            self.current_image = 0
            self.frames = np.zeros((self.N, *cv2.imread(self.dataset_path + "\\" + self.current_video + "\\" + "0.png").shape), dtype=np.uint8)
            # load first N frames
            for i in range(self.N):
                frame = cv2.imread(self.dataset_path + "\\" + self.current_video + "\\" + str(i) + ".png")
                self.frames[i] = frame
            frame = cv2.imread(self.dataset_path + "\\" + self.current_video + "\\" + str(self.N) + ".png")
            return True
    
    def get_frames(self):
        '''
        Return the current frames
        return: list of N frames
        '''
        return self.frames.copy()

    def progress(self):
        '''
        Return the progress of the dataset
        return: float
        '''
        return self.current_N_sequence / self.N_sequences
    
    



if __name__ == "__main__":
    dataset_path = "C:\\projects\\dataset_synthetic_output"
    videos = ["video_0", "video_1","video_2"]
    loader = DatasetLoader(dataset_path, videos, N = 80)
    sequences_array = []
    i = 0
    done = False
    while not done:
        frames = loader.get_frames()
        sequences_array.append(deepcopy(frames))
        done = not loader.load_next_frame()
        print("Progress", loader.progress())
    print("N_sequences", loader.N_sequences)
    print("Number of sequences", len(sequences_array))
    print("End of the dataset")