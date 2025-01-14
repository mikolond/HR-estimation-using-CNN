from extract_face import FaceExtractor
import numpy as np
import csv
import os
import cv2

DEBUG = True

class DatasetCreator:
    def __init__(self, path_in, path_out, flag="ecg-fitness", model_path = 'mediapipe_model\\blaze_face_short_range.tflite'):
        if not os.path.exists(path_in):
            raise FileNotFoundError(f"Input path {path_in} does not exist")
        self.path_in = path_in
        self.path_out = path_out
        self.flag = flag
        self.face_extractor = FaceExtractor(model_path)
        self.cap = None
        self.actual_frame = 0
        self.actual_file = None
        self.fps = None
        self.frame_count = 0
        self.video_out_counter = 84

    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video {path} not found")
    
    def get_video_specs(self):
        if self.cap is None:
            raise ValueError("Video not loaded")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def  load_frame(self):
        if self.cap is None:
            raise ValueError("Video not loaded")
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not ret:
            return None
        self.actual_frame += 1
        return frame

    def unload_video(self):
        self.cap.release()
    
    def create_dataset(self):
        if self.flag == "ecg-fitness":
            self.create_dataset_ecg_fitness()
        elif self.flag == "ecg-fitness-bb":
            self.create_dataset_ecg_fitness_bb()

    def ecg_process_video(self, src_path, dest_path):
        if DEBUG:
            no_face_streak = 0
            no_face_streak_max = 0
            was_face_before = True
        no_face_counter = 0
        # load video
        self.load_video(src_path)
        self.get_video_specs()
        # create folder for video
        os.makedirs(dest_path)
        # read frames
        for i in range(self.frame_count):
            frame = self.load_frame()
            # extract face from frame
            face = self.face_extractor.extract_face(frame)
            if face is None:
                no_face_counter += 1
                if DEBUG:
                    if not was_face_before:
                        no_face_streak += 1
                    was_face_before = False
                continue
            if DEBUG:
                if not was_face_before:
                    if no_face_streak > no_face_streak_max:
                        no_face_streak_max = no_face_streak
                    no_face_streak = 0
                was_face_before = True
            # resize image to 192x128
            resized_face = cv2.resize(face, (128, 192), interpolation=cv2.INTER_NEAREST_EXACT)
            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
            # save face to video_out_folder name the file as frame number
            if face is not None:
                cv2.imwrite(os.path.join(dest_path, f"{i}.png"), resized_face)
        if DEBUG:
            print(f"Max no face streak: {no_face_streak_max}")
        self.unload_video()

        with open(os.path.join(dest_path, 'fps.txt'), 'w') as f:
            f.write(str(self.fps))

        self.video_out_counter += 1
        return no_face_counter
    
    def ecg_load_hr_data(self, path):
        if not os.path.exists(os.path.join(path,"c920.csv")):
            raise FileNotFoundError(f"Path {path}\\c920.csv does not exist")
        if not os.path.exists(os.path.join(path,"viatom-raw.csv")):
            raise FileNotFoundError(f"Path {path}\\viatom-raw.csv does not exist")
        
        with open(os.path.join(path,"c920.csv"), "r") as csv_file:
            time_ref = list(csv.reader(csv_file))
        with open(os.path.join(path,"viatom-raw.csv"), "r") as csv_file:
            hr_data = list(csv.reader(csv_file))

        # find out the index of the HR data ECG HR / PPG HR
        if hr_data[1][2] == -1 and time_ref[1][4] == -1:
            raise ValueError(f"No HR data in viatom-raw.csv in path {path}")
        # choosing ECG HR first because it is more accurate
        if hr_data[1][2] != -1:
            hr_data_index = 2
        else:
            # if ECG HR is not available, choose PPG HR
            hr_data_index = 4

        data = []
        for i in range(len(time_ref)):
            #print("index: ", int(time_ref[i][1])+1, "hr data index: ", hr_data_index, "hr data shape: ", len(hr_data))
            data.append(abs(int(hr_data[int(time_ref[i][1])][hr_data_index])))
        
        return data

    def create_dataset_ecg_fitness(self):
        if not os.path.exists(self.path_out):
            os.makedirs(self.path_out)
        subjects_list = os.listdir(self.path_in)
        for subject in subjects_list:
            folder_list = os.listdir(os.path.join(self.path_in, subject))
            for folder in folder_list:
                # load video c920-1.avi
                # create folder for video based on video out counter
                c920_1_folder = os.path.join(self.path_out, f"video_{self.video_out_counter}")
                c920_path = os.path.join(self.path_in, subject, folder, 'c920-1.avi')
                no_face_counter = self.ecg_process_video(c920_path, c920_1_folder)
                if DEBUG:
                    print(f"Frames without face in {c920_path}: {no_face_counter}")
                if no_face_counter > 700:
                    raise ValueError(f"Too many frames without face in {c920_path}")
                
                # load video c920-2.avi
                c920_2_folder = os.path.join(self.path_out, f"video_{self.video_out_counter}")
                c920_path = os.path.join(self.path_in, subject, folder, 'c920-2.avi')
                no_face_counter += self.ecg_process_video(c920_path, c920_2_folder)
                if DEBUG:
                    print(f"Frames without face in {c920_path}: {no_face_counter}")
                if no_face_counter > 700:
                    raise ValueError(f"Too many frames without face in {c920_path}")
                
                # load c920.csv
                hr_data = self.ecg_load_hr_data(os.path.join(self.path_in, subject, folder))
                # save HR data to video_out_folder for c920-1.avi into txt file
                with open(os.path.join(c920_1_folder, 'hr_data.txt'), 'w') as f:
                    for item in hr_data:
                        f.write(str(item) + "\n")
                # save HR data to video_out_folder for c920-2.avi into txt file
                with open(os.path.join(c920_2_folder, 'hr_data.txt'), 'w') as f:
                    for item in hr_data:
                        f.write(str(item) + "\n")

    def ecg_process_video_bb(self, src_path, dest_path, bb_path):
        bbox = open(bb_path, "r")
        if DEBUG:
            no_face_streak = 0
            no_face_streak_max = 0
            was_face_before = True
        no_face_counter = 0
        # load video
        self.load_video(src_path)
        self.get_video_specs()
        # create folder for video
        os.makedirs(dest_path)
        # read frames
        for i in range(self.frame_count):
            frame = self.load_frame()
            bbox_line = bbox.readline()
            bbox_line = bbox_line.strip().split()
            # extract face from frame
            x_origin, y_origin, bb_width, bb_height = int(float(bbox_line[1])), int(float(bbox_line[2])), int(float(bbox_line[3])), int(float(bbox_line[4]))
            bb_origin = [x_origin, y_origin]
            # desired bb ratio
            bb_ratio = 3/2
            bb_height_new =bb_ratio * bb_width
            height_difference = bb_height_new - bb_height

            pt1 = (int(bb_origin[0]), int(bb_origin[1]-height_difference/2))
            pt2 = (int(bb_origin[0]+bb_width), int(bb_origin[1]+bb_height+height_difference/2))
                    # check if the bounding box is out of the image and shift the bb if it is
            if pt1[0] < 0:
                pt2 = (pt2[0] - pt1[0], pt2[1])
                pt1 = (0, pt1[1])
            if pt1[1] < 0:
                pt2 = (pt2[0], pt2[1] - pt1[1])
                pt1 = (pt1[0], 0)
            if pt2[0] > frame.shape[1]:
                pt1 = (pt1[0] - (pt2[0] - frame.shape[1]), pt1[1])
                pt2 = (frame.shape[1], pt2[1])
            if pt2[1] > frame.shape[0]:
                pt1 = (pt1[0], pt1[1] - (pt2[1] - frame.shape[0]))
                pt2 = (pt2[0], frame.shape[0])

            face = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            if face is None:
                no_face_counter += 1
                if DEBUG:
                    if not was_face_before:
                        no_face_streak += 1
                    was_face_before = False
                continue
            if DEBUG:
                if not was_face_before:
                    if no_face_streak > no_face_streak_max:
                        no_face_streak_max = no_face_streak
                    no_face_streak = 0
                was_face_before = True
            # resize image to 192x128
            resized_face = cv2.resize(face, (128, 192), interpolation=cv2.INTER_NEAREST_EXACT)
            resized_face = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
            # save face to video_out_folder name the file as frame number
            if face is not None:
                cv2.imwrite(os.path.join(dest_path, f"{i}.png"), resized_face)
        if DEBUG:
            print(f"Max no face streak: {no_face_streak_max}")
        self.unload_video()
        bbox.close()

        with open(os.path.join(dest_path, 'fps.txt'), 'w') as f:
            f.write(str(self.fps))

        self.video_out_counter += 1
        return no_face_counter

    def create_dataset_ecg_fitness_bb(self):
        if not os.path.exists(self.path_out):
            os.makedirs(self.path_out)
        subjects_list = os.listdir(self.path_in)
        if "bbox" in subjects_list:
            subjects_list.remove("bbox")
            bbox_path = os.path.join(self.path_in, "bbox")
        else:
            raise FileNotFoundError(f"Folder bbox not found in {self.path_in}")
        for subject in subjects_list:
            folder_list = os.listdir(os.path.join(self.path_in, subject))
            for folder in folder_list:
                # load video c920-1.avi
                # create folder for video based on video out counter
                c920_1_folder = os.path.join(self.path_out, f"video_{self.video_out_counter}")
                c920_1_bb_path = os.path.join(bbox_path, subject, folder, 'c920-1.face')
                c920_path = os.path.join(self.path_in, subject, folder, 'c920-1.avi')
                no_face_counter = self.ecg_process_video_bb(c920_path, c920_1_folder, c920_1_bb_path)
                if DEBUG:
                    print(f"Frames without face in {c920_path}: {no_face_counter}")
                if no_face_counter > 700:
                    raise ValueError(f"Too many frames without face in {c920_path}")
                
                # load video c920-2.avi
                c920_2_folder = os.path.join(self.path_out, f"video_{self.video_out_counter}")
                c920_2_bb_path = os.path.join(bbox_path, subject, folder, 'c920-2.face')
                c920_path = os.path.join(self.path_in, subject, folder, 'c920-2.avi')
                no_face_counter += self.ecg_process_video_bb(c920_path, c920_2_folder, c920_2_bb_path)
                if DEBUG:
                    print(f"Frames without face in {c920_path}: {no_face_counter}")
                if no_face_counter > 700:
                    raise ValueError(f"Too many frames without face in {c920_path}")
                
                # load c920.csv
                hr_data = self.ecg_load_hr_data(os.path.join(self.path_in, subject, folder))
                # save HR data to video_out_folder for c920-1.avi into txt file
                with open(os.path.join(c920_1_folder, 'hr_data.txt'), 'w') as f:
                    for item in hr_data:
                        f.write(str(item) + "\n")
                # save HR data to video_out_folder for c920-2.avi into txt file
                with open(os.path.join(c920_2_folder, 'hr_data.txt'), 'w') as f:
                    for item in hr_data:
                        f.write(str(item) + "\n")


if __name__ == '__main__':
    path_in = "D:\\diplomka\\ecg_fitness\\unzipped"
    path_out = "C:\\projects\\dataset_10_16"
    flag = "ecg-fitness-bb"
    dataset_creator = DatasetCreator(path_in, path_out, flag)
    dataset_creator.create_dataset()

                








    

