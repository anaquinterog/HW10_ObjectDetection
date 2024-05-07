import cv2 
import numpy as np

# Load images
image1 = cv2.imread('/Users/anabi/Documents/GitHub/HW10_ObjectDetection/images/first_image.png')
image2 = cv2.imread('/Users/anabi/Documents/GitHub/HW10_ObjectDetection/images/first_image.png')

# Resize images
image1 = cv2.resize(image1, (0,0), None, 0.4, 0.4)
image2 = cv2.resize(image2, (0,0), None, 0.4, 0.4)

# ORB Detector
orb =  cv2.ORB_create(nfeatures=1000)

# Detect ORB features and compute descriptors.
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Brute Force Matching with k-nearest neighbour calculation
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])

# Draw matches
img3 = cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, good, None, flags=2)

# Display images
cv2.imshow('image1', image1)
cv2.imshow('image2', image2)
cv2.imshow('img3', img3)

# Wait for the 'q' key to be pressed to exit
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all windows
cv2.destroyAllWindows()