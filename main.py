import matplotlib.pyplot as plt
import numpy as np
import string
import math

import cv2

img = cv2.imread('letterR.jpg')

print(img.shape)
# print(img)

rows,cols = img.shape[:2]

# Translate
TR = np.float32([[1,0,100],[0,1,50]])

# Rotate
f = 85;
radian = f / 180 * math.pi;
RT = np.float32([[math.cos(radian), -math.sin(radian), 1], [math.sin(radian), math.cos(radian), 1]])
# RT = cv2.getRotationMatrix2D((cols/2,rows/2), 90, 1)

# Scale
scaleX, scaleY = 1.5, 1
SC = np.float32([[scaleX, 0, 0], [0, scaleY, 0]])

# Affine Transformation
A = np.array([[50,50],[200,50],[50,200]])
B = np.array([[10,100],[200,50],[100,250]])
pts1 = np.float32(A)
pts2 = np.float32(B)
print(A[:,0])

M = cv2.getAffineTransform(pts1, pts2)



dst = cv2.warpAffine(img, M, (cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()



# plt.subplot(121), plt.imshow(img), plt.title('Input'), plt.scatter(A[:,0], A[:,1], color = 'green')
# plt.subplot(122), plt.imshow(dst), plt.title('Output'), plt.scatter(B[:,0], B[:,1], color = 'green')
# plt.show()