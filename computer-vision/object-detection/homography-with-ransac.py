import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

def label_corners(img, corners, radius, colour, thickness):
    corner_image = np.copy(img)  # Create a copy of the input image
    for i in range(corners.shape[0]):
        x = int(corners[i,0])
        y = int(corners[i,1])
        cv2.circle(corner_image, (x, y), radius, colour, thickness) # Draw circle
    return corner_image  # Return the image with detected corners marked

def homography(u2_trans, v2_trans, u_base, v_base):
    
    # number of points
    n = len(u2_trans)

    # for every point, add two rows to the A matrix
    A = np.ones((2*n , 9))
    for i in range(n):
        A[2*i,:] = [u_base[i], v_base[i], 1, 0, 0, 0, -u2_trans[i]*u_base[i], -u2_trans[i]*v_base[i], -u2_trans[i]]
        A[2*i+1,:] = [0, 0, 0, u_base[i], v_base[i], 1, -v2_trans[i]*u_base[i], -v2_trans[i]*v_base[i], -v2_trans[i]]

    # find the eigenvalues of A^T @ A by using SVD decomposition and return the vector corresponding to the smallest eigenvalue
    _, _, V = np.linalg.svd(A)
    H_norm = V[-1].reshape(3, 3)
    
    return H_norm

    # calculate the inverse of the transformation matrix for normalising image coordinates
    T = np.linalg.inv(np.reshape([w+h, 0, w/2, 0, w+h, h/2, 0, 0, 1], (3,3)))
    
    # apply transformation to each point to get normalised coordinates
    newpts = T @ pts.T
    newpts = newpts.T
    return T, newpts

# assumes pts is a N x 3 matrix of x,y,1 coordinates
def normalise(pts, w, h):
    # calculate the inverse of the transformation matrix for normalising image coordinates
    T = np.linalg.inv(np.reshape([w+h, 0, w/2, 0, w+h, h/2, 0, 0, 1], (3,3)))
    
    # apply transformation to each point to get normalised coordinates
    newpts = T @ pts.T
    newpts = newpts.T
    return T, newpts

# assumes input vectors are un-normalised
def homography_w_normalisation(u2_trans, v2_trans, u_base, v_base, img_base, img_trans):
    
    # extract necessary dimensions for normalisation
    h_src, w_src, _ = img_base.shape
    h_dest, w_dest, _ = img_trans.shape

    # create an array of ones with the same size as the input arrays to add to the points to make them homogenous
    ones = np.ones(shape=(len(u_base)))
    
    # create a matrix of homogenous coordinates for the base and transformed points
    pts1 = np.array([u_base,v_base,ones]).T
    pts2 = np.array([u2_trans,v2_trans,ones]).T
    
    # normalise the points in the source and destination images
    T1, pts1 = normalise(pts1,w_src, h_src)
    T2, pts2 = normalise(pts2, w_dest, h_dest)

    # compute the homography using normalised points
    H_normalised = homography(pts2[:,0], pts2[:,1], pts1[:,0], pts1[:,1])

    # denormalise the computed homography
    H = np.linalg.inv(T2) @ H_normalised @ T1
    return H

def matching_features(img1, img2, ratio_test_threshold=0.7):

    # create a SIFT object
    sift = cv2.SIFT_create()

    # extract key features and descriptors from the two images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # create a Brute Force (BF) Matcher object to find matches returning the 2 best matches for every kp from img1
    # when compared to all kp in img2. the result is a img.shape[0] x 2 matrix containing distances (similarities)
    bf = cv2.BFMatcher()
    knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # if the distances of the two best matches are two similar, then there are more than 1 feature in img2 that is
    # similar to the query kp from img1 which means its probably not a good match. therefore we use a ratio test to 
    # ensure they are dissimilar enough. helps filter out ambiguous matches 
    matches = []
    for m1, m2 in knn_matches:
        if (m1.distance / m2.distance) < ratio_test_threshold:
            matches.append(m1)

    # for each match, find the corresponding keypoint from the base and the trans image, and save the coordiantes
    # of these matching points into an array
    base_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]) 
    trans_points = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # randomly sample half of the points using seed 42
    sample_size = len(base_points) // 2
    indices = np.random.choice(base_points.shape[0], size=sample_size, replace=False)
    sample_base = base_points[indices, :]
    sample_trans = trans_points[indices, :]

    # np.save('src_pts.npy', sample_base)
    # np.save('dst_pts.npy', sample_trans)

    # draw circles around the detected matches 
    circle_base = label_corners(img1, sample_base, 1, (255,0,0), 2)
    circle_trans = label_corners(img2, sample_trans, 1, (255,0,0), 2)

    # plot the circled images
    _, ax = plt.subplots(1,2)
    ax[0].imshow(circle_base)
    ax[1].imshow(circle_trans)
    ax[0].set_title("mountain1.jpg")
    ax[1].set_title("mountain2.jpg")
    plt.show()

    # convert to homogenous coordinates
    base_points = np.hstack((base_points, np.ones(shape=(base_points.shape[0],1))))
    trans_points = np.hstack((trans_points, np.ones(shape=(trans_points.shape[0],1))))

    return base_points, trans_points


# function to find the homography between two sets of corresponding points, using RANSAC for robustness
def homography_w_normalisation_ransac(pts1, pts2, img_base, img_trans, distance_thres=300, num_points=4, iterations = 500):
    
    # get the width and height of both source and destination images for normalisation
    h_src, w_src, _ = img_base.shape
    h_dest, w_dest, _ = img_trans.shape

    # these are the source and destination points we're going to be working with
    pts1 = base_points
    pts2 = trans_points

    # we'll keep track of the best homography we find and how many inliers it has
    best_homography = 0
    most_inliers = -1
    n = len(pts1)

    # ransac loop - we're going to do this a bunch of times
    for i in range(iterations):
        # at each iteration, randomly sample 'num_points' from the base and transformed point sets
        indices = np.random.choice(n, num_points, replace=False)
        pts1_sample = pts1[indices]
        pts2_sample = pts2[indices]

        # calculate the homography for these points using the previous function
        H = homography_w_normalisation(pts2_sample[:,0], pts2_sample[:,1], pts1_sample[:,0], pts1_sample[:,1], img_base, img_trans)

        # apply the homography to all points in the base image
        pts2_transformed = (H @ pts1.T).T        

        distances = []
        # for each transformed point, calculate the distance from its corresponding point in the destination image
        for i in range(len(pts2_transformed)):
            x = pts2[i,0]
            y = pts2[i,1]
            x_prime = pts2_transformed[i,0]
            y_prime = pts2_transformed[i,1]
            distance = np.sqrt((x - x_prime) ** 2 + (y - y_prime) ** 2)
            distances.append(distance)

        # find all the inliers (points where the distance is less than our threshold)
        inliers = np.array(distances) < distance_thres

        # if we've found a homography with more inliers than before, update our best homography
        if (np.sum(inliers) > most_inliers):
            most_inliers = np.sum(inliers)
            best_homography = H

    # after all iterations, return the best homography we found and how many inliers it had
    return best_homography, most_inliers

def plot_stiched_images(img1, img2, H, ax, i, j):
    img1_warped = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    img1_warped[:, :img2.shape[1]] = img2
    ax[i,j].imshow(img1_warped)
    return ax

# variation in threshold
def warp_and_plot(pts1, pts2, img1, img2, thresholds):

    num_thresholds = len(thresholds)
    _, axes = plt.subplots(num_thresholds, 1, figsize=(15, 15))

    Hs = []
    for i, threshold in enumerate(thresholds):
        # Modify the homography matrix based on the threshold
        H, inliers = homography_w_normalisation_ransac(pts1, pts2, img1, img2, threshold, 4, 1000)
        Hs.append(H)
        img1_warped = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        img1_warped[:, :img2.shape[1]] = img2

        # Plot the result
        axes[i].imshow(img1_warped)
        axes[i].set_title(f'Distance Threshold = {threshold}', fontsize=12)

    plt.tight_layout()
    plt.show()
    return Hs, thresholds

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("img1_path", help="Path to first image")
    parser.add_argument("img2_path", help="Path to second image")
    parser.add_argument("--ratio_test_threshold", type=float, default=0.7, help="Threshold for ratio test in feature matching")
    parser.add_argument("--distance_threshold", type=int, default=500, help="Distance threshold for RANSAC")
    parser.add_argument("--num_points", type=int, default=50, help="Number of points for RANSAC")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations for RANSAC")

    args = parser.parse_args()

    img1 = cv2.imread(args.img1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread(args.img2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    ax[0].set_title("Image 1")
    ax[1].set_title("Image 2")
    plt.show()

    base_points, trans_points = matching_features(img1,img2,args.ratio_test_threshold)

    H = homography_w_normalisation(trans_points[:,0], trans_points[:,1], base_points[:,0], base_points[:,1], img1, img2)
    img1_warped = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    img1_warped[0:img2.shape[0], 0:img2.shape[1]] = img2
    fig,ax = plt.subplots()
    ax.imshow(img1_warped)
    ax.set_title("Homography Before RANSAC")

    # load in the previous established points
    pts1 = np.load('../data/src_pts.npy')
    pts2 = np.load('../data/dst_pts.npy')

    # calling the function with some base and transformed points, an image pair, distance threshold, number of points and iterations
    H_no_ransac, inliers = homography_w_normalisation_ransac(pts1, pts2, img1, img2, args.distance_threshold, args.num_points, args.iterations)
    img1_warped = cv2.warpPerspective(img1, H_no_ransac, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    img1_warped[0:img2.shape[0], 0:img2.shape[1]] = img2
    fig,ax = plt.subplots()
    ax.imshow(img1_warped)
    ax.set_title("Homography After RANSAC")

    plt.show()
