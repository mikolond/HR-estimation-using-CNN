from model_extractor import Extractor
# from my_extractor import Extractor
from dataset_loader import DatasetLoader
import torch
import numpy as np
import os
import csv


def process_video(model, extractor_dataset_path, video, estimator_dataset_path, N):
    # create dataset_loader just for the 1 video
    # reate csv file in estimator_dataset_path with the same name as the video
    # until video is not done
        # load N sequence
        # get extractor output of the sequence
        # get hr list from dataset_loader
        # save the the frame number, extractor output, hr for each frame in the csv file
    dataset_loader = DatasetLoader(extractor_dataset_path, [video], N, N)
    csv_file_path = os.path.join(estimator_dataset_path, f"{video}.csv")

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_number', 'extractor_output', 'hr'])
        dataset_done = False
        sequence_count = 0
        with torch.no_grad():
            while not dataset_done:
                sequence = dataset_loader.get_sequence()
                hr_data = dataset_loader.get_hr_list()
                x = torch.tensor(sequence.reshape(N,192,128,3).transpose(0,3,1,2)).float().to(device)
                extractor_output = model(x)
                # save to csv file N rows
                for i in range(N):
                    csv_writer.writerow([i + N * sequence_count, extractor_output[i].item(), hr_data[i]])
                sequence_count += 1
                dataset_done = not dataset_loader.next_sequence()

if __name__ == "__main__":
    # load extractor dataset
    dataset_path = os.path.join("datasets", "dataset_synthetic")
    N = 600
    videos_list = os.listdir(dataset_path)
    # keep only the names that contains "video"
    videos_list = [video for video in videos_list if "video" in video]


    # load extractor model
    weights_path = os.path.join("output","weights","model_epoch_2.pth")
    device_id = input("Enter the device number: ")
    if torch.cuda.is_available():
        device = torch.device("cuda:" + device_id)
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")

    extractor = Extractor()
    extractor.eval()
    extractor.to(device)
    extractor.load_state_dict(torch.load(weights_path, map_location=device))

    # estimaotr dataset path
    estimator_dataset_path = os.path.join("datasets", "estimator_synthetic")

    for video in videos_list:
        print("Processing video " + video)
        process_video(extractor, dataset_path, video, estimator_dataset_path, N)
        print("Video " + video + " done")

