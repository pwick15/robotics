import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

CANNY_THRESHOLD_LOWER = 400
CANNY_THRESHOLD_UPPER = 150
RADIUS_LOWER = 10
RADIUS_UPPER = 35
RADIUS_STEP = 2
NUM_THETA = 45
VOTES_THRESHOLD = 0.5
EDGE_THRESHOLD = 10
KSIZE = 50
NMS_THRESHOLD = 0.5

gray_image = cv2.imread("computer-vision/data/coins.png")

if len(gray_image.shape) > 2:
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

# FIXME remove these plots and put them all at the end
plt.imshow(gray_image, cmap='gray')
plt.show()

# FIXME remove these plots and put them all at the end
edges = cv2.Canny(gray_image, CANNY_THRESHOLD_LOWER, CANNY_THRESHOLD_UPPER)
plt.imshow(edges, cmap='gray')
plt.show()


def draw_circles(image, circle_params, color, thickness):
    # Convert the grayscale image to a 3-channel image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw circles on the image
    for params in circle_params:
        center = (params[0], params[1])
        radius = params[2]
        cv2.circle(image_rgb, center, radius, color, thickness)
    return image_rgb

def hough_circle_transform(edges, radius_range, radius_step, num_thetas, votes_threshold, edge_threshold):
    
    # define rows and cols
    rows = edges.shape[0]
    cols = edges.shape[1]

    # define necessary parameters
    radii = range(radius_range[0], radius_range[1], radius_step)
    num_radius = len(radii)

    # initialize accumulator
    accumulator = np.zeros(shape = (rows,cols,num_radius))

    # iterate over each pixel in the edge image
    for y in range(rows):
        for x in range(cols):
            if edges[y, x] > edge_threshold:  # if this pixel is an edge pixel
                
                # for each possible radius
                for r_idx, r in enumerate(radii):

                    # for each possible value of theta
                    for theta in np.linspace(0, 2*np.pi, num_thetas):
                        a = int(x - r * np.cos(theta))
                        b = int(y - r * np.sin(theta))
                        
                        # Check if the parameters are within valid range
                        if 0 <= a < accumulator.shape[0] and 0 <= b < accumulator.shape[1]:
                            accumulator[a, b, r_idx] += 1

    # Find circles
    output_circles = []
    max_votes = np.max(accumulator)
    for a in range(accumulator.shape[0]):
        for b in range(accumulator.shape[1]):
            for r_idx in range(accumulator.shape[2]):
                if accumulator[a, b, r_idx] / max_votes > votes_threshold:
                    output_circles.append((a, b, r_idx*radius_step + radius_range[0]))

    return accumulator, output_circles, np.array(range(RADIUS_LOWER, RADIUS_UPPER, RADIUS_STEP))


acc, out, radii = hough_circle_transform(edges = edges, 
                                 radius_range=(RADIUS_LOWER,RADIUS_UPPER), 
                                 radius_step = RADIUS_STEP, 
                                 num_thetas = NUM_THETA, 
                                 votes_threshold = VOTES_THRESHOLD,
                                 edge_threshold=EDGE_THRESHOLD)

circled = draw_circles(gray_image, out, (0,0,255), 1)
plt.imshow(cv2.cvtColor(circled, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


def compress3dto2d(all_votes, all_radii):
    # create 2D array with same size as acc
    radii = np.zeros(shape=(all_votes.shape[0], all_votes.shape[1])) 
    votes = np.zeros(shape=(all_votes.shape[0], all_votes.shape[1])) 

    for a in range(all_votes.shape[0]):
        for b in range(all_votes.shape[1]):

            # extract the votes across all radii for this particular a and b value
            list_radii = all_votes[a,b,:]
            most_votes_idx = np.argmax(list_radii)
            most_votes = list_radii[most_votes_idx]
            votes[a,b] = most_votes
            radii[a,b] = all_radii[most_votes_idx]

    return radii, votes


def nms(votes, radii, ksize, thresh):

    # Add padding around the R matrix
    padding = ksize // 2 # padding size
    votes = np.pad(votes, ((padding, padding), (padding, padding)), 'constant')  # Add padding
    max_votes = np.max(votes)
    rows, cols = votes.shape  # get dimensions
    good_circles = []  # output list for corner points

    # Loop through all points of the accumulator, checking if central point is greater than all points in the window
    for i in range(padding, rows - padding):  # Iterate through rows
        for j in range(padding, cols - padding):  # Iterate through cols
            
            # Define window region for NMS
            region = votes[i - padding: i + padding + 1, j - padding : j + padding + 1]
            
            # Find center coordinates
            x_ctr, y_ctr = i, j
            
            # Suppress points below threshold
            if votes[x_ctr, y_ctr] != np.max(region):
                votes[x_ctr, y_ctr] = 0
                
            # Add point to output if above threshold
            elif votes[x_ctr, y_ctr] > thresh * max_votes:
                radius = radii[x_ctr - padding, y_ctr - padding]  # Access the original 'radii' array
                good_circles.append([x_ctr - padding, y_ctr - padding, int(radius)])  # Subtract padding to return to original coordinates
    
    return good_circles


radii, votes = compress3dto2d(acc, radii)
good_circles = nms(votes, radii, KSIZE, NMS_THRESHOLD)


circled = draw_circles(gray_image, good_circles, (0,0,255), 1)
plt.imshow(cv2.cvtColor(circled, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()