import numpy as np
import math
import os
import csv



f_s = 30
def create_freq_signal(f,f_s,length, amplitude, noise_amplitude, noise_frequency):
    signal = np.zeros(f_s * length)
    c = 2 * np.pi * f / 60 / f_s 
    c_noise = 2 * np.pi * noise_frequency / 60 / f_s
    for i in range(f_s * length):
        signal[i] = math.sin(i*c) * amplitude
        signal[i] += math.sin(i*c_noise) * noise_amplitude
    return signal

def create_file(signal, path, file_name, f):
    file = open(os.path.join(path, file_name + ".csv"), "w")
    file.write("frame_number,extractor_output,hr\n")
    for i in range(len(signal)):
        file.write(f"{i},{signal[i]},{f}\n")

def create_dataset(path, f_min, f_max, f_s, length):
    if not os.path.exists(path):
        os.makedirs(path)
    for f in range(f_min, f_max):
        signal = create_freq_signal(f, f_s, length, 3, 0, np.random.uniform(0, 20))
        create_file(signal, path, str(f), f)
        print(f"File {f} created")

    # create data.csv file with all filename in 0 groups randomly organised
    arr = np.arange(f_min, f_max) 
    groups = 10
    in_the_group = len(arr) // groups
    filepath = os.path.join(path, "data.csv")

    with open(filepath, "w") as file:
        while True:
            group = []
            for i in range(in_the_group):

                arr_len = len(arr)
                if arr_len == 0:
                    break
                idx = np.random.randint(arr_len)  # Random index
                chosen = arr[idx]                  # Selected value
                group.append(chosen)  # Add to group

                # Remove the chosen element from the array
                arr = np.delete(arr, idx)
            if len(group) > 0:
                file.write(",".join([str(x) for x in group]) + "\n")
            if len(group) == 0:
                break
    print("data.csv created")




if __name__ == "__main__":
    path = "datasets/freq_preset"
    f_min = 30
    f_max = 251
    f_s = 30
    length = 60
    create_dataset(path, f_min, f_max, f_s, length)