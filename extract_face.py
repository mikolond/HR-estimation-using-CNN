import numpy as np
import mediapipe as mp
import time
import cv2
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceExtractor():
    def __init__(self, model_path = 'mediapipe_model\\blaze_face_short_range.tflite'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        base_options = python.BaseOptions(model_asset_path='mediapipe_model\\blaze_face_short_range.tflite')
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)

    def extract_face(self, img):
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = self.detector.detect(image)
        if len(detection_result.detections) == 0:
            return None
        bb = detection_result.detections[0].bounding_box
        bb_origin = [bb.origin_x, bb.origin_y]
        bb_width = bb.width
        bb_height = bb.height
        pt1 = (int(bb_origin[0] * img.shape[1]), int(bb_origin[1] * img.shape[0]))
        pt2 = (int((bb_origin[0] + bb_width) * img.shape[1]), int((bb_origin[1] + bb_height) * img.shape[0]))

        # check if the bounding box is out of the image and shift the bb if it is
        if pt1[0] < 0:
            pt2 = (pt2[0] - pt1[0], pt2[1])
            pt1 = (0, pt1[1])
        if pt1[1] < 0:
            pt2 = (pt2[0], pt2[1] - pt1[1])
            pt1 = (pt1[0], 0)
        if pt2[0] > img.shape[1]:
            pt1 = (pt1[0] - (pt2[0] - img.shape[1]), pt1[1])
            pt2 = (img.shape[1], pt2[1])
        if pt2[1] > img.shape[0]:
            pt1 = (pt1[0], pt1[1] - (pt2[1] - img.shape[0]))
            pt2 = (pt2[0], img.shape[0])

        img_face = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        return img_face