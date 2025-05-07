import os
import cv2
import numpy as np

def create_synthetic_dataset(videos_folder_path, dataset_folder_path):
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create synthetic dataset")
    parser.add_argument("videos_folder_path", type=str, help="Path to the folder with videos")
    parser.add_argument("dataset_folder_path", type=str, help="Path to the folder where the dataset will be saved")
    args = parser.parse_args()
    videos_folder_path = args.videos_folder_path
    dataset_folder_path = args.dataset_folder_path
    create_synthetic_dataset(videos_folder_path, dataset_folder_path)