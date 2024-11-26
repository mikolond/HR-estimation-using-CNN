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
img = cv2.imread('faces1.jpg')
# image = mp.Image.create_from_file('face1.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the image to a MediaPipe Image format.
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)


# image = mp.Image(format=mp.ImageFormat.SRGB, data=img)

# STEP 4: Detect faces in the input image.
start_time = time.time()
detection_result = detector.detect(image)
print("resiults: ", detection_result)
print("Time taken: ", time.time() - start_time)

bb = detection_result.detections[0].bounding_box
bb_origin = [bb.origin_x, bb.origin_y]
bb_width = bb.width
bb_height = bb.height

kp = detection_result.detections[0].keypoints
print("Keypoints: ", kp)

print("Bounding box: ", bb)

# desired bb ratio
bb_ratio = 3/2
bb_height_new =bb_ratio * bb_width
print("New height: ", bb_height_new)
print("Old height: ", bb_height)
print("Width: ", bb_width)
height_difference = bb_height_new - bb_height
print("Height difference: ", height_difference)

# create rectangle points
pt1 = (int(bb_origin[0]), int(bb_origin[1]-height_difference/2))
pt2 = (int(bb_origin[0]+bb_width), int(bb_origin[1]+bb_height+height_difference/2))

# get new image just from the bounding box
img_face = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]

cv2.imshow('face', img_face)
cv2.waitKey(0)

# draw bounding box using opencv
cv2.rectangle(img, pt1,pt2,
              (0, 255, 0), 2)

# draw keypoints using opencv
img_width = img.shape[1]
img_height = img.shape[0]
for i,point in enumerate(kp):
    print(f'point:{point}, i:{i}')
    cv2.circle(img, (int(point.x * img_width), int(point.y * img_height)), 2, (0, 255, 0), 10)
    cv2.putText(img, str(i), (int(point.x * img_width), int(point.y * img_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


#show image
cv2.imshow('image', img)
cv2.waitKey(0)