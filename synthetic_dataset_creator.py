import os
import cv2
import numpy as np


videos_folder_path = "test_videos\\"
dataset_folder_path = "C:\\projects\\dataset_synthetic_output\\"

# load all filenames in the videos folder
filenames = os.listdir(videos_folder_path)

# find all .avi files
video_filenames = [filename for filename in filenames if filename.endswith(".avi")]

video_count = 0

for video in video_filenames:

    # create a folder for each video
    os.makedirs(dataset_folder_path + 'video_' + str(video_count))
    # load video
    cap = cv2.VideoCapture(videos_folder_path + video)
    # get video parameters
    frames_c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame count",frames_c)
    f_s = int(cap.get(cv2.CAP_PROP_FPS))
    # load N frames
    for i in range(frames_c):
        ret, frame = cap.read()
        if not ret:
            break

        # save frame
        path = dataset_folder_path + "video_" + str(video_count) + "\\"
        cv2.imwrite(path + str(i)+".png", frame)
    cap.release()
    # copy the txt to the dataset folder
    f = open(videos_folder_path + video.replace(".avi", ".txt"), "r")
    lines = f.readlines()
    f.close()
    f = open(dataset_folder_path + 'video_' + str(video_count) +'\\hr_data.txt', "w")
    for line in lines:
        f.write(line)
    f.close()

    f = open(dataset_folder_path + 'video_' + str(video_count) +'\\fps'+ '.txt', "w")
    f.write(str(f_s))
    f.close()

    video_count += 1