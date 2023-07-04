import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import convolve


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image processing tasks.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    return parser.parse_args()

def addGausNoise(img, mean, sigma):
    gaussian_noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = np.clip(img + gaussian_noise, 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(1,2, figsize=(12,4))

    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(noisy_img, cmap='gray')
    axs[1].set_title('Gaussian Noise Added')
    fig.subplots_adjust(wspace=0.4) # Adjust the space between the subplots

    plt.show()

    return noisy_img

def histogramChange(img, noised):
    hist_before = cv2.calcHist([img], [0], None, [256], [0,256])
    hist_after = cv2.calcHist([noised], [0], None, [256], [0,256])

    fig, axs = plt.subplots(1,2,figsize=(15,4))
    axs[0].plot(hist_before)
    axs[0].set_title('PDF before Gaussian Noise Added')
    axs[1].plot(hist_after)
    axs[1].set_title('PDF before Gaussian Noise Added')

    plt.show()


# Since a Gaussian filter has the property of being seperable, we can produce a 7x7 Gaussian
# filter as the outer product of two 1D 7 dimensional Gaussian kernel. 

def gaussianKernel(ksize = 7, sigma = 1):
    
    # Generate the 1D Gaussian kernel
    kernel = cv2.getGaussianKernel(ksize, sigma)
    
    # Compute the 2D Gaussian kernel by taking the outer product of the 1D kernel
    kernel_2d = kernel * kernel.T

    # Normalize the kernel so its elements sum to 1
    kernel_2d = kernel_2d / kernel_2d.sum()
    
    return kernel_2d

def my_Gauss_filter(noisy, ksize, sigma):
    # assumes we are doing valid mode convolution and that the outer region will keep the same pixels as before
    # assumes the input is a grayscale image

    # extract the rows and columns
    rows = noisy.shape[0]
    cols = noisy.shape[1]

    # create the template for the new image
    denoised = noisy.copy()

    # produce the 2x2 kernel
    kernel = gaussianKernel(ksize, sigma)

    # define the padding which will distinguish the valid mode from the outer region
    padding = ksize // 2

    # for the pixels in the valid mode, perform the filtering
    for i in range(padding, rows - padding):
        for j in range(padding, cols - padding):
            # define the image region
            image = noisy[i - padding: i + padding + 1, j - padding : j + padding + 1]

            # perform the correlation
            denoised[i,j] = np.sum(image * kernel)

    fig, axs = plt.subplots(1,2,figsize=(15,4))
    axs[0].imshow(noisy, cmap='gray')
    axs[0].set_title('Noisy Image')
    axs[1].imshow(denoised, cmap='gray')
    axs[1].set_title('My Gaussian Filtering')
    fig.subplots_adjust(wspace=0.4) # Adjust the space between the subplots

    plt.show()



if __name__ == "__main__":
    args = parse_arguments()
    image_path = args.image_path
 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    noised = addGausNoise(img, 0, 15)

    histogramChange(img, noised)

    my_Gauss_filter(noised, 7, 5)


    

    