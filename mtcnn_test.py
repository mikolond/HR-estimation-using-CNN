from mtcnn import MTCNN
import cv2
import numpy as np
from mtcnn.utils.images import load_image


mtcnn = MTCNN()



img = load_image('faces1.jpg')

resoults = mtcnn.detect_faces(img)
print(resoults[0])

# print(img_cropped.shape) # shape (3,160,160)
# # show the cropped image

# # make shape (160,160,3)
# img_show = img_cropped.permute(1, 2, 0).numpy()
# print(img_show.shape)

# img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)

# cv2.imshow("cropped", img_show)
# cv2.waitKey(0)


