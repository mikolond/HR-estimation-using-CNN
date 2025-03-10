import os
import cv2
import random
import numpy as np


DEBUG = False


class DatasetLoader:
    def __init__(self, dataset_path, videos = None, N=100, step_size=1,augmentation=False):
        self.dataset_path = dataset_path
        if videos is None:
            self.videos = os.listdir(dataset_path)
        else:
            self.videos = videos
        self.N = N
        self.step_size = step_size
        self.last_sequence_loaded = False
        self.augmentation = augmentation
        self.flip = True

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
                if self.augmentation:
                    # rotate the image
                    M = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), np.radians(self.augmentation_angle), 1)
                    frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
                    # add random shade
                    frame = np.clip(frame + self.augmentation_color, 0, 255)
                    # flip the image
                    if self.flip:
                        frame = cv2.flip(frame, 1)
                    if DEBUG:
                        # convert frame to a supported depth
                        frame_to_show = frame.astype(np.uint8)
                        # show the frame
                        print("frame shape", frame_to_show.shape)
                        cv2.imshow("frame", frame_to_show)
                        cv2.waitKey(0)
                self.frames = np.roll(self.frames, -1, axis=0)
                self.frames[-1:] = frame
                self.current_image += 1

                # delete first hr data in current hr data
                self.current_hr_data = np.roll(self.current_hr_data, -1)
                # load new hr data
                self.current_hr_data[-1] = int(self.hr_data.readline())
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
        if self.augmentation:
            self.set_augmentation()
        self.current_video = self.videos[self.current_video_idx]
        images_count = len(os.listdir(os.path.join(self.dataset_path, self.current_video))) - 2
        self.current_sequences = (images_count - self.N) // self.step_size + 1
        self.current_video_frames_count = images_count
        self.current_image = 0
        self.frames = np.zeros((self.N, *cv2.imread(os.path.join(self.dataset_path, self.current_video, "0.png")).shape), dtype=np.uint8)
        # load the next hr data
        self.hr_data = open(os.path.join(self.dataset_path, self.current_video, "hr_data.txt"), "r")
        self.current_hr_data = np.array([int(self.hr_data.readline()) for i in range(self.N)])

        # load first N frames
        for i in range(self.N):
            frame = cv2.imread(os.path.join(self.dataset_path, self.current_video, str(i) + ".png"))
            if self.augmentation:
                # rotate the image
                M = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), self.augmentation_angle, 1)
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
                # add random color
                frame = np.clip(frame + self.augmentation_color, 0, 255)

                if self.flip:
                    frame = cv2.flip(frame, 1)
                if DEBUG:
                    # convert frame to a supported depth
                    frame_to_show = frame.astype(np.uint8)
                    # show the frame
                    print("frame shape", frame_to_show.shape)
                    cv2.imshow("frame", frame_to_show)
                    cv2.waitKey(0)
            self.frames[i] = frame

    def set_augmentation(self):
        # generate agumentation parameters
        # random angle
        angle = 30
        self.augmentation_angle = random.uniform(-angle,angle)
        # random color
        color = 10
        self.augmentation_color = np.random.randint(-color, color) * np.ones((1,1,3), dtype=np.uint8)
    
    def reset(self):
        '''
        Reset the dataset loader 
        if augmentation is enabled, the frames will be augmented (random color added and rotated by a small angle)
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

        # set augmentation
        self.flip = not self.flip



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
        return float(np.mean(self.current_hr_data))
    
    def get_hr_list(self):
        '''
        Return the current hr list
        return: list of int
        '''
        return self.current_hr_data
    
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


