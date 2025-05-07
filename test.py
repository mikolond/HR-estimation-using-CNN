import cv2
from face_extractor import FaceExtractor
import numpy as np
import time

# Open the default camera
cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
# choose a codec that supports uncompressed video


face_extractor = FaceExtractor()
faces_buffer = np.zeros((300,192,128,3))
i = 0
start_time = time.time()
while True:
    ret, frame = cam.read()
    capture_time = time.time()
    frame_time = capture_time - start_time
    start_time = capture_time


    print("fps:", 1/frame_time, end="\r")

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()