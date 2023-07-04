import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import progressbar

title_size = 8

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image processing tasks.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    return parser.parse_args()

def my_Sobel_filter3x3(image):
    rows, cols = image.shape
    filtered = np.zeros(shape=image.shape)
    Gx = np.zeros(shape=image.shape)
    Gy = np.zeros(shape=image.shape)
    angle = np.zeros(shape=image.shape)

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    padding = 1

    # Set up the progress bar
    bar = progressbar.ProgressBar(maxval=rows-padding, \
        widgets=[progressbar.Bar('#', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i in range(padding, rows - padding):
        bar.update(i)  # Update the progress bar
        for j in range(padding, cols - padding):
            region = image[i - padding: i + padding + 1, j - padding: j + padding + 1]
            x = np.sum(region * kernel_x)
            y = np.sum(region * kernel_y)

            Gx[i, j] = x
            Gy[i, j] = y
            filtered[i, j] = np.sqrt(x ** 2 + y ** 2)

            if x == 0:
                angle[i, j] = 90
            else:
                angle[i, j] = np.arctan(y / x)
    bar.finish()  # Finish the progress bar

    plt.figure(figsize=(15, 15))
    plt.imshow(filtered, cmap='gray')
    plt.title('My Sobel Edge Detector')
    plt.show()

if __name__ == "__main__":
    args = parse_arguments()
    image_path = args.image_path

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    my_Sobel_filter3x3(img)
