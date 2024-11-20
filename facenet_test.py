from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2

mtcnn = MTCNN()

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

from PIL import Image

img = Image.open('faces1.jpg')

img_cropped = mtcnn(img)

print(img_cropped.shape) # shape (3,160,160)
# show the cropped image

# make shape (160,160,3)
img_show = img_cropped.permute(1, 2, 0).numpy()
print(img_show.shape)

img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)

cv2.imshow("cropped", img_show)
cv2.waitKey(0)


