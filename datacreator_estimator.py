from Models.extractor_model import Extractor
from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
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
    dataset_loader = DatasetLoader(extractor_dataset_path, [video], N, N, augmentation=False)
    csv_file_path = os.path.join(estimator_dataset_path, f"{video}.csv")
    if not os.path.exists(estimator_dataset_path):
        os.makedirs(estimator_dataset_path)
        # if file already exists skip the file
    if os.path.exists(csv_file_path):
        print(f"File {csv_file_path} already exists, skipping...")
        return


    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_number', 'extractor_output', 'hr'])
        dataset_done = False
        sequence_count = 0
        next_return = N
        with torch.no_grad():
            while not dataset_done:
                sequence = dataset_loader.get_sequence()
                hr_data = dataset_loader.get_hr_list()

                sequence = sequence[N-next_return:]
                x = torch.tensor(sequence.reshape(next_return,192,128,3).transpose(0,3,1,2)).float().to(device)
                extractor_output = model(x)
                # save to csv file N rows
                for i in range(next_return):
                    csv_writer.writerow([i + N * sequence_count, extractor_output[i].item(), hr_data[i]])
                sequence_count += 1
                next_return = dataset_loader.next_sequence()
                if next_return is None:
                    dataset_done = True
                
    


if __name__ == "__main__":
    # load extractor dataset
    dataset_path = os.path.join("datasets", "dataset_ecg_fitness")
    N = 800
    videos_list = os.listdir(dataset_path)
    # delete all files keep only folders
    for video in videos_list:
        if not os.path.isdir(os.path.join(dataset_path, video)):
            videos_list.remove(video)


    # load extractor model
    weights_path = os.path.join("output","weights","ecg_decreasing", "best_extractor_weights.pth")
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
    estimator_dataset_path = os.path.join("datasets", "estimator_ecg_decreasing")

    for video in videos_list:
        print("Processing video " + video)
        process_video(extractor, dataset_path, video, estimator_dataset_path, N)
        print("Video " + video + " done")

    # copy data.csv from dataset_path to estimator_dataset_path
    os.system(f"cp {os.path.join(dataset_path, 'data.csv')} {os.path.join(estimator_dataset_path, 'data.csv')}")
    print("All videos done")

