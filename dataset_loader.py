import os
import cv2
import random
from copy import deepcopy
import numpy as np



class DatasetLoader:
    def __init__(self, dataset_path, videos, N=100, step_size=1):
        self.dataset_path = dataset_path
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

    def next_sequence(self):
        '''
        Load the next frames according to the step_size
        return: frame
        '''
        self.current_N_sequence += 1
        frames_to_load = [i for i in range(self.current_image + self.N, self.current_image + self.N + self.step_size)]
        if frames_to_load[-1] < self.current_video_frames_count:
            for frame_id in frames_to_load:
                frame = cv2.imread(os.path.join(self.dataset_path, self.current_video, str(frame_id) + ".png"))
                self.frames = np.roll(self.frames, -1, axis=0)
                self.frames[-1:] = frame
                self.current_image += 1

                # delete first hr data in current hr data
                self.current_hr_data = np.roll(self.current_hr_data, -1)
                # load new hr data
                self.current_hr_data[-1] = int(self.hr_data.readline())

            self.current_hr = np.mean(self.current_hr_data)
            return True
        else:
            # load the next video
            self.current_video_idx += 1
            if self.current_video_idx >= len(self.videos):
                return False
            self.load_next_video()
            return True

    def load_next_video(self):
        '''
        Load the next video and initialize frames and hr data
        '''

        self.current_video = self.videos[self.current_video_idx]
        images_count = len(os.listdir(os.path.join(self.dataset_path, self.current_video))) - 2
        self.current_sequences = (images_count - self.N) // self.step_size + 1
        self.current_video_frames_count = images_count
        self.current_image = 0
        self.frames = np.zeros((self.N, *cv2.imread(os.path.join(self.dataset_path, self.current_video, "0.png")).shape), dtype=np.uint8)
        # load the next hr data
        self.hr_data = open(os.path.join(self.dataset_path, self.current_video, "hr_data.txt"), "r")
        self.current_hr_data = np.array([int(self.hr_data.readline()) for i in range(self.N)])
        self.current_hr = np.mean(self.current_hr_data)

        # load first N frames
        for i in range(self.N):
            frame = cv2.imread(os.path.join(self.dataset_path, self.current_video, str(i) + ".png"))
            self.frames[i] = frame
    
    def reset(self):
        '''
        Reset the dataset loader
        '''
        self.N_sequences = 0
        for video in self.videos:
            images_count = len(os.listdir(os.path.join(self.dataset_path, video))) - 2
            self.N_sequences += (images_count - self.N) // self.step_size + 1
        
        # shuffle the videos
        self.current_video_idx = 0
        random.shuffle(self.videos)
        self.load_next_video()
        self.current_N_sequence = 0

        self.fps_data = open(os.path.join(self.dataset_path, self.current_video, "fps.txt"), "r")
        self.current_fps = float((self.fps_data.readline()))

    def get_sequence(self):
        '''
        Return the current frames
        return: list of N frames
        '''
        return self.frames.copy()

    def get_hr(self):
        '''
        Return the current hr
        return: float
        '''
        return float(self.current_hr)
    
    def get_fps(self):
        '''
        Return the current fps
        return: int
        '''
        return self.current_fps

    def progress(self):
        '''
        Return the progress of the dataset
        return: float
        '''
        return [self.current_N_sequence , self.N_sequences]



if __name__ == "__main__":
    dataset_path = "C:\\projects\\dataset_synthetic_output"
    videos = ["video_0", "video_1"]
    loader = DatasetLoader(dataset_path, ["video_0","video_1"], N = 88, step_size=1)
    sequences_array = []
    i = 0
    done = False
    while not done:
        frames = loader.get_sequence()
        print("Sequence", i, "Shape", frames.shape)
        hr = loader.get_hr()
        fps = loader.get_fps()
        sequences_array.append(deepcopy(frames))
        done = not loader.next_sequence()
        i+=1
        print("Progress", loader.progress(), "HR", hr, "FPS", fps)
    print("N_sequences", loader.N_sequences)
    print("Number of sequences", len(sequences_array))
    print("End of the dataset")
    loader.reset()
    sequences_array = []
    i = 0
    done = False
    while not done:
        frames = loader.get_sequence()
        print("Sequence", i, "Shape", frames.shape)
        hr = loader.get_hr()
        fps = loader.get_fps()
        sequences_array.append(deepcopy(frames))
        done = not loader.next_sequence()
        i+=1
        print("Progress", loader.progress(), "HR", hr, "FPS", fps)
    print("N_sequences", loader.N_sequences)
    print("Number of sequences", len(sequences_array))
    print("End of the dataset")
