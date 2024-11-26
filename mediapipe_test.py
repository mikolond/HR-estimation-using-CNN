# import mediapipe as mp
# import numpy as np
# BaseOptions = mp.tasks.BaseOptions
# FaceDetector = mp.tasks.vision.FaceDetector
# FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
# VisionRunningMode = mp.tasks.vision.RunningMode
# import cv2
# from PIL import Image
# # from mediapipe.tasks import python
# # from mediapipe.tasks.python import vision

# model_path = 'mediapipe_model/blaze_face_short_range.tflite'
# options = FaceDetectorOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.IMAGE)

# with FaceDetector.create_from_options(options) as detector:
#     # img = cv2.imread('face1.jpg')
#     # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),dtype=np.uint8)
#     img = Image.open('face1.jpg')
#     img_np = np.array([img], dtype=np.uint8)
#     mp_image = mp.Image(format=mp.ImageFormat.SRGB, data=img_np)
#     face_detector_result = detector.detect(mp_image)
#     print(face_detector_result)

# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
import time
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='mediapipe_model/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file('face1.jpg')

# STEP 4: Detect faces in the input image.
start_time = time.time()
detection_result = detector.detect(image)
print("resiults: ", detection_result)
print("Time taken: ", time.time() - start_time)

bb = detection_result.detections[0].bounding_box
bb_center = [bb.origin_x, bb.origin_y]
bb_width = bb.width
bb_height = bb.height

kp = detection_result.detections[0].keypoints
print("Keypoints: ", kp)

print("Bounding box: ", bb)

img = cv2.imread('face1.jpg')

# draw bounding box using opencv
cv2.rectangle(img, (int(bb_center[0]), int(bb_center[1])),
              (int((bb_center[0] + bb_width)), int((bb_center[1] + bb_height))),
              (0, 255, 0), 2)

#show image
cv2.imshow('image', img)
cv2.waitKey(0)