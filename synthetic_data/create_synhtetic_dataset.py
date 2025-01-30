from synthetic_data.synthetic_videos_generator import create_video
from synthetic_data.synthetic_dataset_creator import create_synthetic_dataset
import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic dataset of 100 synthetic videos with different constant HR frequencies and 10 videos with random changing HR frequencies")
    parser.add_argument("dataset_folder", type=str, help="Path to the folder where the dataset will be saved", default="dataset/")

    save_path = "test_videos/"
    dataset_path = parser.parse_args().dataset_folder

    # train dataset videos path
    videos_path_train = os.path.join(save_path, "train")
    # valid dataset videos path
    videos_path_valid = os.path.join(save_path, "valid")

    # create directories
    if not os.path.exists(videos_path_train):
        os.makedirs(videos_path_train)
    if not os.path.exists(videos_path_valid):
        os.makedirs(videos_path_valid)

    freq_start = 50
    end_freq = 200

    all_freqs = np.arange(freq_start, end_freq)
    shuffled_freqs = np.random.shuffle(all_freqs)
    train_freqs = all_freqs[:int(len(all_freqs)*0.7)]
    valid_freqs = all_freqs[int(len(all_freqs)*0.7):]

    # generate train videos
    # videos with constant frequencies
    for freq in train_freqs:
        create_video(videos_path_train, str(freq), freq, 30, 60, 0, 0, 0, 1)

    # videos with changing frequencies
    create_video(videos_path_train, "random0", 0, 30, 60, 60, 90, 0.8, 1)
    create_video(videos_path_train, "random1", 0, 30, 60, 110, 200, 0.5, 1)
    create_video(videos_path_train, "random2", 0, 30, 60, 110, 130, 0.5, 1)
    create_video(videos_path_train, "random3", 0, 30, 60, 40, 100, 0.3, 1)
    create_video(videos_path_train, "random4", 0, 30, 60, 50, 120, 0.7, 1)
    create_video(videos_path_train, "random5", 0, 30, 60, 40, 200, 0.8, 1)
    create_video(videos_path_train, "random6", 0, 30, 60, 150, 200, 0.3, 1)

    # generate valid videos
    # videos with constant frequencies
    for freq in valid_freqs:
        create_video(videos_path_valid, str(freq), freq, 30, 60, 0, 0, 0, 1)

    # videos with changing frequencies
    create_video(videos_path_valid, "random7", 0, 30, 60, 40, 60, 0.2, 1)
    create_video(videos_path_valid, "random8", 0, 30, 60, 60, 80, 0.5, 1)
    create_video(videos_path_valid, "random9", 0, 30, 60, 50, 80, 0.8, 1)

    # create synthetic train dataset
    dataset_path_train = os.path.join(dataset_path, "train")
    dataset_path_valid = os.path.join(dataset_path, "valid")
    # create directories
    if not os.path.exists(dataset_path_train):
        os.makedirs(dataset_path_train)
    if not os.path.exists(dataset_path_valid):
        os.makedirs(dataset_path_valid)

    create_synthetic_dataset(videos_path_train, dataset_path_train)
    create_synthetic_dataset(videos_path_valid, dataset_path_valid)