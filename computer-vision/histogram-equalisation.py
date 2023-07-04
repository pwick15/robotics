import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage import color


title_size = 8


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image processing tasks.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    return parser.parse_args()

### assumes the image is a single channel image ###
### assume the image is an np.array ###


def histogramEqualisation(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Mask of non-zero (i.e. valid) cumulative histogram values
    cdf_m = np.ma.masked_equal(cdf_normalized, 0)
    # Normalization: Subtract minimum (and convert to np.uint8)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # Wherever there was a masked value, place a 0 in the result
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Use the cumulative distribution function as a lookup table to get the output image
    equalised_img = cdf[img]

    return equalised_img


def equalisationResults(img, img_name, fig, axs):

    equalised = histogramEqualisation(img)

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    axs[0, 0].plot(hist)
    axs[0, 0].set_title(
        '{} Channel PDF after Equalisation'.format(img_name), fontsize=title_size)

    hist = cv2.calcHist([equalised], [0], None, [256], [0, 256])
    axs[1, 0].plot(hist)
    axs[1, 0].set_title(
        '{} Channel PDF after Equalisation'.format(img_name), fontsize=title_size)

    axs[0, 1].imshow(img, cmap='gray')
    axs[0, 1].set_title(
        '{} Image before Equalisation'.format(img_name), fontsize=title_size)

    axs[1, 1].imshow(equalised, cmap='gray')
    axs[1, 1].set_title(
        '{} Image after Equalisation'.format(img_name), fontsize=title_size)

    return axs


if __name__ == "__main__":
    args = parse_arguments()
    image_path = args.image_path

    img = cv2.imread(image_path)
    titles = ["B", "G", "R"]

    for i in range(3):
        fig, axs = plt.subplots(2,2, figsize=(10, 7))
        axs = equalisationResults(img[:,:,i], titles[i], fig, axs)
        plt.show()

    