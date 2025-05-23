from Datasets_handlers.Extractor.dataset_loader import DatasetLoader
import torch
import numpy as np
import os
import csv
from utils import load_model_class

class DatasetCreator:
    def __init__(self, weights_path, device, source_path, dest_path, N, extractor_model_path, augmentation=False):
        Extractor = load_model_class(extractor_model_path, "Extractor")
        self.model = Extractor().to(device)
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            raise FileNotFoundError(f"Extractor weights:{weights_path} not found")
        self.model.eval()
        self.dataset_path = source_path
        self.estimator_dataset_path = dest_path
        if not os.path.exists(self.estimator_dataset_path):
            os.makedirs(self.estimator_dataset_path)
        self.N = N
        self.augmentation = augmentation
        self.device = device
    
    def create_dataset(self):
        videos_list = os.listdir(self.dataset_path)
        # delete data.csv file from video_list
        for video in videos_list:
            if not os.path.isdir(os.path.join(self.dataset_path, video)):
                videos_list.remove(video)

        for video in videos_list:
            print("Processing video " + video)
            process_video(self.model, self.dataset_path, video, self.estimator_dataset_path, self.N, self.device)
            print("Video " + video + " done")
        
        # copy data.csv from dataset_path to estimator_dataset_path
        os.system(f"cp {os.path.join(self.dataset_path, 'data.csv')} {os.path.join(self.estimator_dataset_path, 'data.csv')}")
        print("All videos done")





def process_video(model, extractor_dataset_path, video, estimator_dataset_path, N, device):
    # create dataset_loader just for the 1 video
    # reate csv file in estimator_dataset_path with the same name as the video
    # until video is not done
        # load N sequence
        # get extractor output of the sequence
        # get hr list from dataset_loader
        # save the the frame number, extractor output, hr for each frame in the csv file
    dataset_loader = DatasetLoader(extractor_dataset_path, [video], N, N)
    csv_file_path = os.path.join(estimator_dataset_path, f"{video}.csv")
    if not os.path.exists(estimator_dataset_path):
        os.makedirs(estimator_dataset_path)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_number', 'extractor_output', 'hr'])
        dataset_done = False
        sequence_count = 0
        frames_to_load = N
        with torch.no_grad():
            while not dataset_done:
                sequence = dataset_loader.get_sequence()
                hr_data = dataset_loader.get_hr_list()
                x = torch.tensor(sequence.reshape(N,192,128,3).transpose(0,3,1,2)).float().to(device)
                extractor_output = model(x).reshape(N)
                # save to csv file N rows
                if frames_to_load == N:
                    for i in range(N):
                        csv_writer.writerow([i + N * sequence_count, extractor_output[i].item(), hr_data[i]])
                else:
                    # save only the last frames_to_load rows
                    for i in range(frames_to_load):
                        csv_writer.writerow([i + N * sequence_count, extractor_output[i].item(), hr_data[i]])
                sequence_count += 1
                frames_to_load = dataset_loader.next_sequence()
                if frames_to_load is None:
                    dataset_done = True