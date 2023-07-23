import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read images
# ensure images are grayscale
img1 = cv2.imread('/Users/punjayawickramasinghe/Downloads/IMG_0383.jpg', 0)
img2 = cv2.imread('/Users/punjayawickramasinghe/Downloads/IMG_0384.jpg', 0)

sf = 0.7
img1 = cv2.resize(img1, (0,0), fx=sf, fy=sf) 
img2 = cv2.resize(img2, (0,0), fx=sf, fy=sf) 


# Feature extraction (SIFT)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Feature matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test to find good matches
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print("Feature matching: complete")

# Compute Fundamental matrix
pts1 = np.int32([kp1[m[0].queryIdx].pt for m in good])
pts2 = np.int32([kp2[m[0].trainIdx].pt for m in good])

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
print("Fundamental matrix calculation: complete")

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

K = np.array([[3.20512987e+03, 0.00000000e+00, 1.99443897e+03],
              [0.00000000e+00, 3.17391061e+03, 1.41309060e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Compute Essential matrix
E = K.T @ F @ K
print("Essential matrix calculation: complete")

# Compute the projection matrix for the two cameras
# For the first camera we can choose P = K[I|0] (Camera at the origin)
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

# Second camera matrix P' = K[R|t]
# Compute 'R' and 't' from Essential matrix decomposition
U, D, Vt = np.linalg.svd(E)
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
print("Projection matrix calculation: complete")

# There are four possible combinations to choose from
R = U @ W @ Vt
t = U[:, 2]
P2 = K @ np.hstack((R, t.reshape(-1, 1)))

# Triangulation
point_4D_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
point_3D = point_4D_hom / np.tile(point_4D_hom[-1, :], (4, 1))
point_3D = point_3D[:3, :].T
print("Triangulation: complete")

# New visualization code...
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_3D[:, 0], point_3D[:, 1], point_3D[:, 2])
plt.show()
