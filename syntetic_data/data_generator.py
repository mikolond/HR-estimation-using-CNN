import cv2
import numpy as np
import math

# Define the codec and create VideoWriter object
# Here 'XVID' is the codec, and the output is saved as an AVI file
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
out = cv2.VideoWriter('test_videos/output.avi', fourcc, 20.0, (640, 480))  # Filename, codec, FPS, frame size

c = 3
amplitude = 50
for i in range(200):
    static_array = np.array([[[100 + math.sin(i/c)*amplitude, 100 + math.sin(i/c)*amplitude, 0]] * 640] * 480, dtype=np.uint8) 
    static_image = cv2.cvtColor(static_array, cv2.COLOR_BGR2RGB)
    out.write(static_image)
    print("Frame ", i)

# Release everything when done
out.release()
print("Video saved!")
