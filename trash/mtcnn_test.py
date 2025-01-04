from facenet_pytorch import MTCNN
import cv2
import numpy as np
import time
import torch
from PIL import Image


mtcnn = MTCNN(device='cuda:0')

img = cv2.imread('face1.jpg')
print("shape: ", img.shape)
img2 = cv2.imread('faces1.jpg')
width = img.shape[1]
height = img.shape[0]
cv2.resize(img,(int(height/3), int(width/3)), interpolation=cv2.INTER_NEAREST_EXACT)

resoults = mtcnn(img2)
start = time.time()
resoults = mtcnn(img)
print("Time: ", time.time()-start)
res_numpy = resoults.permute(1, 2, 0).numpy()

print("type: ", type(res_numpy))
print("shape: ", res_numpy.shape)
cv2.imshow("cropped", res_numpy)
cv2.waitKey(0)


# resoults = mtcnn.detect_faces(img2)
# print("Time: ", time.time()-start)

# print(img_cropped.shape) # shape (3,160,160)
# # show the cropped image

# # make shape (160,160,3)
# img_show = img_cropped.permute(1, 2, 0).numpy()
# print(img_show.shape)

# img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)

# cv2.imshow("cropped", img_show)
# cv2.waitKey(0)


