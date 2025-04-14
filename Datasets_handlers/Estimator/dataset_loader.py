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
                raise Exception("Video path does not exist:", os.path.join(self.dataset_path, video))
            
        self.reset()
        
        
    def reset(self):
        '''
        Reset the loader to the first video
        '''
        self.N_sequences = 0
        self.current_video_index = 0
        random.shuffle(self.videos)

        self.load_next_video()


    def load_next_video(self):
        '''
        Load the next video
        '''
        self.current_video = self.videos[self.current_video_index]
        csv_filepath = os.path.join(self.dataset_path, self.current_video)
        with open(csv_filepath, 'r') as file:
            reader = csv.reader(file)
            self.current_video_frames = list(reader)
        # skip the header 
        self.current_video_frames = self.current_video_frames[1:]
        self.current_row = 0

        self.current_video_frames_count = len(self.current_video_frames)

        # load first N frames
        N_data = self.current_video_frames[self.current_row:self.current_row + self.N]
        # use header [frame number, extractor output, ground truth]
        self.current_N_sequence = np.array(N_data)[:,1].astype(np.float64)
        self.current_hr_data = np.array(N_data)[:,2].astype(np.int64)

    def next_sequence(self):
        '''
        Load the next frames according to the step_size
        return: end of file (boolean)
        '''
        self.N_sequences += 1
        last_frame_idx = self.current_row + self.N + self.step_size
        if last_frame_idx < self.current_video_frames_count:
            N_data = self.current_video_frames[self.current_row + self.step_size:last_frame_idx]
            self.current_N_sequence = np.array(N_data)[:,1].astype(np.float64)
            self.current_hr_data = np.array(N_data)[:,2].astype(np.int64)
            self.current_row += self.step_size
            return True
        elif last_frame_idx - self.N < self.current_video_frames_count:
            difference = last_frame_idx - self.current_video_frames_count
            N_data = self.current_video_frames[self.current_row + self.step_size - difference:self.current_video_frames_count]
            self.current_N_sequence = np.array(N_data)[:,1].astype(np.float64)
            self.current_hr_data = np.array(N_data)[:,2].astype(np.int64)
            self.current_row += self.step_size + self.N
            return True
        else:
            # load the next video
            self.current_video_index += 1
            if self.current_video_index >= len(self.videos):
                self.last_sequence_loaded = True
                return False
            self.load_next_video()
            return True
        
    def get_sequence(self):
        return self.current_N_sequence, np.median(self.current_hr_data)
    
    def get_progress(self):
        return self.current_video_index, len(self.videos)


if __name__ == "__main__":
    dataset_path = os.path.join("datasets", "estimator_synthetic")
    loader = EstimatorDatasetLoader(dataset_path, N = 100, step_size = 100)

    data = loader.get_sequence()
    print(data[0].shape)

    for i in range(19):
        sequence, hr = loader.get_sequence()
        print(sequence)
        print(hr)
        print(sequence.shape)
        print("Progress:", loader.get_progress())
        loader_finished = not loader.next_sequence()
        print("Finished:", loader_finished)
        # loader.reset()
        # print("Reset")



