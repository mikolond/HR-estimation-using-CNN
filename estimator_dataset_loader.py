import os
import csv
import random
import numpy as np


DEBUG = False


class EstimatorDatasetLoader:
    def __init__(self, dataset_path, videos = None, N=100, step_size=1):
        self.dataset_path = dataset_path
        if videos is None:
            self.videos = os.listdir(dataset_path)
        else:
            self.videos = videos
        self.N = N
        self.step_size = step_size
        self.last_sequence_loaded = False

        # check if the dataset path exists
        if not os.path.exists(self.dataset_path):
            raise Exception("Dataset path does not exist")
        
        # check if the videos are in the dataset path
        for video in self.videos:
            if not os.path.exists(os.path.join(self.dataset_path, video)):
                raise Exception("Video path does not exist")
            
        self.reset()
        
        
    def reset(self):
        '''
        Reset the loader to the first video
        '''
        self.N_sequences = 0
        self.current_video_index = 0

        self.load_next_video()

    def load_next_video(self):
        '''
        Load the next video
        '''
        self.current_video = self.videos[self.current_video_index]
        csv_filepath = os.path.join(self.dataset_path, self.current_video)
        with open(csv_filepath, 'r') as file:
            self.reader = csv.reader(file)
            self.current_video_frames = list(self.reader)
        # skip the header 
        self.current_video_frames = self.current_video_frames[1:]

        self.current_video_frames_count = len(self.current_video_frames)


