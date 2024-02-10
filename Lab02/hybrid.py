import sys
import cv2
import numpy as np

def cross_correlation(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    
    #   Get local copy of img
    image = img
    
    #   Extract image dimensions height and width
    image_height, image_width = image.shape[:2]
    
    #   Extract kernel dimensions, assuming that the kernel is m x n
    kernel_height, kernel_width = kernel.shape

    #   Initialize number of image channels
    image_channels = 0
    
    #   Check if the image is grayscale or RGB
    if len(image.shape) == 3:             #     Cover the case where the image is colored
        image_channels = image.shape[2]
    else:                                 #     Cover the case where the image is greyscale
        image_channels = 1
        image = image[:, :, np.newaxis]   #     Convert grayscale image to 3-channel (for consistency)

    #   Initialize blank image for result of operation
    processed_image = np.zeros_like(image)
                
    #   Pad image with zeroes
    padded_image = np.zeros((image_height+kernel_height-1,image_width+kernel_width-1, image_channels))
    for c in range(image_channels):
        for i in range(image_height):
            for j in range(image_width):
                padded_image[i + int((kernel_height - 1) / 2), j + int((kernel_width - 1) / 2), c] = image[i, j, c]  #  Copy image to padded array

    #   Loop for computing cross-correlation with kernel
    for i in range(image_height):
        for j in range(image_width):
            for c in range(image_channels):
                #   Extract the window from the padded image for the current color channel
                window = padded_image[i : i + kernel_height, j : j + kernel_width, c]
                #   Perform element-wise multiplication between padded image and kernel
                processed_image[i, j, c] = np.sum(window * kernel)

    return processed_image

def convolution(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    
    return cross_correlation(img, np.flip(kernel))

def gaussian_blur(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it   
        with an image results in a Gaussian-blurred image.
    '''
    kernel = np.zeros((height, width))

    #   Grab coordinates of the center of the kernel
    center_x = width // 2
    center_y = height // 2
    
    #   Generate Gaussian blur kernel
    for i in range(height):
        for j in range(width):
            #   Calculate Gaussian function value at each point
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((i - center_y)**2 + (j - center_x)**2) / (2 * sigma**2))

    #   Normalize kernel values
    kernel = kernel / np.sum(kernel)

    return kernel

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return convolution(img, gaussian_blur(sigma, size, size))


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img - low_pass(img, sigma, size)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)