import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceExtractor:
    def __init__(self,input_size=(640,480)):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], alllowed_modules=['detection'])
        self.app.prepare(ctx_id=0, det_size=input_size)

        self.previous_bb = None
        self.previous_bb_move_vector = None

    def extract_face(self, img):
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detection_result = self.app.get(img)
        if len(detection_result) == 0 and self.previous_bb is None:
            return None
        elif len(detection_result) == 0 and self.previous_bb is not None:
            bb = [self.previous_bb[0] + self.previous_bb_move_vector[0], self.previous_bb[1] + self.previous_bb_move_vector[1],
                    self.previous_bb[2] + self.previous_bb_move_vector[0], self.previous_bb[3] + self.previous_bb_move_vector[1]].astype(int)
        else:
            bb = detection_result[0].bbox.astype(int)
            bb_origin = [bb[0], bb[1]]
            bb_width = bb[2] - bb[0]
            bb_height = bb[3] - bb[1]
            if self.previous_bb is not None:
                self.previous_bb_move_vector = [bb_origin[0] - self.previous_bb[0], bb_origin[1] - self.previous_bb[1]]

        self.previous_bb = bb
        # desired bb ratio
        bb_ratio = 3/2
        bb_height_new =bb_ratio * bb_width
        height_difference = bb_height_new - bb_height

        pt1 = (int(bb_origin[0]), int(bb_origin[1]-height_difference/2))
        pt2 = (int(bb_origin[0]+bb_width), int(bb_origin[1]+bb_height+height_difference/2))

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


if __name__ == '__main__':
    face_extractor = FaceExtractor()
    img = cv2.imread('face.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_face = face_extractor.extract_face(img_rgb)
    if img_face is None:
        print("No face detected")
        exit()
    print()
    cv2.imwrite('face_extracted.jpg', cv2.cvtColor(img_face, cv2.COLOR_RGB2BGR))
    # cv2.imshow('face', cv2.cvtColor(img_face,cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()