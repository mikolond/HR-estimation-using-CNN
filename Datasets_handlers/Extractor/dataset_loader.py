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
        self.new_video = False
        self.augmentation = augmentation
        self.flip = True

        # check if the dataset path exists
        if not os.path.exists(self.dataset_path):
            raise Exception("Dataset path does not exist")
        
        # check if the videos are in the dataset path
        for video in self.videos:
            if not os.path.exists(os.path.join(self.dataset_path, video)):
                raise Exception("Video path does not exist:", os.path.join(self.dataset_path, video))
            
        self.reset()

    def next_sequence(self):
        '''
        Load the next frames according to the step_size
        return: frame
        '''
        self.new_video = False
        self.current_N_sequence += 1
        if self.current_image + self.step_size + self.N <= self.current_video_frames_count:
            # if whole sequence can be loaded
            frames_to_load = np.arange(self.current_image + self.step_size, self.current_image + self.N + self.step_size)
            # find out if any of the frames are already loaded
            # common_frames = np.intersect1d(frames_to_load, self.loaded_frames)
            if self.N > self.step_size:
                common_frames_length = self.N - self.step_size
            else:
                common_frames_length = 0
            # print("common frames", common_frames)
            # print("common frames", common_frames)
            if common_frames_length > 0:
                # if some of the frames are already loaded keep the last common_frames_length frames
                self.frames = np.roll(self.frames,-common_frames_length, axis=0)
                frames_to_load = frames_to_load[common_frames_length:]
        
            # load the next hr data
            indexes_to_load = np.arange(self.current_image + self.step_size, self.current_image + self.step_size + self.N)
            self.current_hr_data = self.hr_data[indexes_to_load]
        

        elif self.current_image + self.step_size < self.current_video_frames_count:
            # if the last sequence can be loaded ( but not fully) load just the rest of the frames
            frames_to_load = np.arange(self.current_image + self.step_size, self.current_video_frames_count)

            if self.N > self.step_size:
                common_frames_length = self.N - self.step_size
            else:
                common_frames_length = 0
            # print("common frames", common_frames)
            # print("common frames", common_frames)
            if common_frames_length > 0:
                # if some of the frames are already loaded keep the last common_frames_length frames
                self.frames = np.roll(self.frames,-common_frames_length, axis=0)
                frames_to_load = frames_to_load[common_frames_length:]

            # load the next hr data
            indexes_to_load = np.arange(self.current_image + self.step_size, self.current_video_frames_count)
            self.current_hr_data = self.hr_data[indexes_to_load]
        else:
            # load the next video
            self.current_video_idx += 1
            if self.current_video_idx >= len(self.videos):
                return None
            return self.load_next_video()
        
        # load the rest of the frames frames
        for frame_id in frames_to_load:
            frame = cv2.imread(os.path.join(self.dataset_path, self.current_video, str(frame_id) + ".png"))
            if self.augmentation:
                # rotate the image
                M = cv2.getRotationMatrix2D((frame.shape[1]//2, frame.shape[0]//2), np.radians(self.augmentation_angle), 1)
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
                # add random shade
                frame = np.clip(frame + self.augmentation_color, 0, 255)
                # flip the image
                if random.choice([True, False]):
                    frame = cv2.flip(frame, 1)
                if DEBUG:
                    # convert frame to a supported depth
                    frame_to_show = frame.astype(np.uint8)
                    # show the frame
                    cv2.imshow("frame", frame_to_show)
                    cv2.waitKey(0)
                # translate the image randomly
                translation = 10
                M = np.float32([[1,0,random.randint(-translation, translation)],[0,1,random.randint(-translation, translation)]])
                frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
            self.frames = np.roll(self.frames, -1, axis=0)
            self.frames[-1:] = frame
        self.current_image += self.step_size
        self.loaded_frames = np.concatenate((self.loaded_frames[self.step_size:], frames_to_load))

    
        return len(frames_to_load)
    


    def load_next_video(self):
        '''
        Load the next video and initialize frames and hr data
        '''
        self.new_video = True
        if self.augmentation:
            self.set_augmentation()
        self.current_video = self.videos[self.current_video_idx]
        images_count = len(os.listdir(os.path.join(self.dataset_path, self.current_video))) - 2
        self.current_sequences = (images_count - self.N) // self.step_size + 1
        self.current_video_frames_count = images_count
        self.current_image = 0
        self.frames = np.zeros((self.N, *cv2.imread(os.path.join(self.dataset_path, self.current_video, "0.png")).shape), dtype=np.uint8)
        # load the next hr data
        hr_data_file = open(os.path.join(self.dataset_path, self.current_video, "hr_data.txt"), "r")
        self.hr_data = np.array([int(line.strip()) for line in hr_data_file])
        self.current_hr_data = self.hr_data[:self.N]

        # load first N frames
        self.loaded_frames = np.arange(self.N)
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
        return self.N

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

        self.N_sequences = 1
        for video in self.videos:
            images_count = len(os.listdir(os.path.join(self.dataset_path, video))) - 2
            self.N_sequences += (images_count - self.N) // self.step_size + 1
            if (images_count - self.N) % self.step_size != 0:
                self.N_sequences += 1  
        
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
        return float(np.median(self.current_hr_data))
    
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

    def get_progress(self):
        '''
        Return the progress of the dataset
        return: float
        '''
        return [self.current_N_sequence , self.N_sequences]





if __name__ == "__main__":
    dataset_path = "datasets/dataset_synthetic"
    videos = ["video_1", "video_2"]
    import time
    start = time.time()


    dataset_loader = DatasetLoader(dataset_path, videos, N=300, step_size=100,augmentation=True)

    while True:
        sequence = dataset_loader.get_sequence()
        hr = dataset_loader.get_hr()
        hr_list = dataset_loader.get_hr_list()
        progress = dataset_loader.get_progress()
        print("progress", progress)
        dataset_done = dataset_loader.next_sequence()
        print("dataset_done", dataset_done)
        if dataset_done is None:
            break
    dataset_loader.reset()

