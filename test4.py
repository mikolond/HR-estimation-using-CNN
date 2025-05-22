import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
y = x**2
custom_x_ticks = [0, 2, 4, 7, 9]
custom_x_labels = ['A', 'B', 'C', 'D', 'E']

plt.plot(x, y)
plt.xticks(custom_x_ticks, custom_x_labels)
plt.xlabel("Custom X-axis Labels")
plt.ylabel("Y-axis")
plt.title("Plot with Custom X-axis Ticks and Labels")
plt.grid(True)
plt.savefig("custom_x_ticks.png")