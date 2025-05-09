import os
import cv2
import random
import numpy as np


DEBUG = False
if DEBUG:
    import time


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
        self.varying_offset = False
        self.offset = 10

    def set_varying_offset(self, offset):
        self.offset = offset
        self.varying_offset = True

    def augment_frame(self, frame):
        '''
        Augment the frame by rotating and zooming it
        '''
        zoom = self.augmentation_zoom
        angle = self.augmentation_angle
        tx = self.augmentation_translation_x
        ty = self.augmentation_translation_y
        # rotate and zoom image
        (h, w) = frame.shape[:2]
        center = (w / 2, h / 2)

        # Get rotation + zoom matrix
        M = cv2.getRotationMatrix2D(center, angle, 1/zoom)

        # Add translation to the matrix (last column)
        M[0, 2] += tx
        M[1, 2] += ty

        # Apply combined transformation
        frame = cv2.warpAffine(
            frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
            )     
        # add random shade
        frame = np.clip(frame + self.augmentation_color, 0, 255)
        # flip the image
        if self.flip:
            frame = cv2.flip(frame, 1)
        # add random zoom
        if DEBUG:
            # convert frame to a supported depth
            frame_to_show = frame.astype(np.uint8)
            # show the frame
            cv2.imwrite("frame.png", frame_to_show)
            time.sleep(1)
        return frame

    def next_sequence(self):
        '''
        Load the next frames according to the step_size
        return: frame
        '''
        if self.varying_offset:
            self.step_size = random.randint(1, self.offset)
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
                frame = self.augment_frame(frame)
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
                frame = self.augment_frame(frame)
            self.frames[i] = frame
        return self.N

    def set_augmentation(self):
        # generate agumentation parameters
        # random angle
        angle = 45
        self.augmentation_angle = np.random.uniform(-angle, angle)
        # random color
        color = 10
        self.augmentation_color = np.random.uniform(-color, color,3)
        # random translation
        translation = 20
        self.augmentation_translation_x = np.random.randint(-translation, translation)
        self.augmentation_translation_y = np.random.randint(-translation, translation)
        # random zoom
        zoom_in = 0.3
        zoom_out = 0.1
        self.augmentation_zoom = np.random.uniform(1+zoom_out, 1-zoom_in)
        # flip
        self.flip = random.choice([True, False])
    
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
    dataset_path = "datasets/dataset_ecg_fitness"
    videos = ["video_1", "video_2"]
    import time
    start = time.time()


    dataset_loader = DatasetLoader(dataset_path, videos, N=300, step_size=100,augmentation=True)
    start = time.time()
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
    print("time", time.time() - start)
    dataset_loader.reset()

