import mediapipe as mp
import numpy as np
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
import cv2
from PIL import Image
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

model_path = 'mediapipe_model/blaze_face_short_range.tflite'
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with FaceDetector.create_from_options(options) as detector:
    # img = cv2.imread('face1.jpg')
    # img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),dtype=np.uint8)
    img = Image.open('face1.jpg')
    img_np = np.array([img], dtype=np.uint8)
    mp_image = mp.Image(format=mp.ImageFormat.SRGB, data=img_np)
    face_detector_result = detector.detect(mp_image)
    print(face_detector_result)
