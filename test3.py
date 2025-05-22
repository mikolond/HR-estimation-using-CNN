import csv
import os
from matplotlib import pyplot as plt
import numpy as np

path = os.path.join("datasets","final_exp1_pure_new_model_split1", "06-03.csv")
sequence_length = 300
with open(path, 'r') as file:
    reader = csv.reader(file)
    # read first row
    header = next(reader)
    # read all rows
    data = []
    for row in reader:
        data.append(row)

# convert to numpy array
data = np.array(data)
# convert to float
data = data.astype(float)
# get 2nd row
row = data[:,1]
# plot first 300 values
data_to_plot = row[:sequence_length]
print(data_to_plot)
print("shape", data_to_plot.shape)
fig = plt.figure(figsize=(10, 5))

x_ticks = np.arange(0, sequence_length+2, 30)
x_labels = np.arange(0, 11, 1)
print(x_ticks)
print(x_labels)
plt.plot(data_to_plot, label='HR Signal')
plt.xticks(x_ticks, x_labels, fontsize=14)  # Increase x-axis tick label font size
plt.yticks(fontsize=14)                     # Increase y-axis tick label font size
plt.title('Extracted rPPG signal', fontsize=16) # Increase title font size
plt.xlabel('Time [s]', fontsize=14)          # Increase x-axis label font size
plt.ylabel('Amplitude', fontsize=14)         # Increase y-axis label font size
plt.legend(fontsize=12)                     # Increase legend font size
plt.savefig("pure_improv_rppg.png",bbox_inches='tight')
plt.close(fig)