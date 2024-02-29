import cv2
import numpy as np
from scipy.signal import convolve2d as convolve

#   Define the binomial 5-tap filter
kernel = (1.0/256) * np.array([[1, 4, 6, 4, 1],[4, 16, 24, 16, 4],[6, 24, 36, 24, 6],[4, 16, 24, 16, 4],[1, 4, 6, 4, 1]])

def interpolate(img):
    """
    Interpolates an image with upsampling rate r = 2.

    :param img: input image with 3 channels (RGB)
    :return: the interpolated image with upsampling rate r = 2
    """
    interpolated_image = np.zeros((2 * img.shape[0], 2 * img.shape[1], 3))
    
    channels = img.shape[2] if len(img.shape) > 2 else 1
    for channel in range(channels):
        # Upsample each channel
        interpolated_image[::2, ::2, channel] = img[:, :, channel]
        # Blur and quadruple kernel area, since we interpolated by 4x
        interpolated_image[:, :, channel] = convolve(interpolated_image[:, :, channel], 4 * kernel, mode='same')
        
    # interpolated_image[::2, ::2] = img[:, :]
    #     # Blur and quadruple kernel area, since we interpolated by 4x
    # interpolated_image[:, :] = convolve(interpolated_image[:, :], 4 * kernel, mode='same')
        
    return interpolated_image.astype(np.uint8)

def decimate(img):
    """
    Decimates an image with downsampling rate r = 2.

    :param img: input image with 3 channels (RGB)
    :return: the decimated image with downsampling rate r = 2
    """
    decimated_image = np.zeros((img.shape[0]//2, img.shape[1]//2, 3))
    
    channels = img.shape[2] if len(img.shape) > 2 else 1
    for channel in range(channels):
        # Blur and decimate each channel
        decimated_image[:, :, channel] = convolve(img[:, :, channel], kernel, mode='same')[::2, ::2]
    
    # decimated_image[:, :] = convolve(img[:, :], kernel, mode='same')[::2, ::2]
    
    return decimated_image.astype(np.uint8)

def construct_gaussian_pyramid(image):
    """
        Constructs a Gaussian pyramid to max depth.
        Parameters:
                image: the original image (i.e. base of the pyramid)
        Returns:
                gaussian_pyramid: A sequence of Gaussian pyramids to max depth.
    """

    #   Gaussian pyramid base is the original image
    gaussian_pyramid = [image, ]

    #   Construct max-depth gaussian pyramid by decimating image at each level until max depth is reached
    while image.shape[0] >= 2 and image.shape[1] >= 2:
        image = decimate(image)
        gaussian_pyramid.append(image)

    return gaussian_pyramid

def construct_pyramids(image):
    """
        Constructs a Laplacian pyramid to max depth and its corresponding Gaussian pyramid to max depth - 1.
        Parameters:
                image: the original image (i.e. base of the pyramid)
        Returns:
                laplacian_pyramid: A sequence of Laplacian pyramids to max depth.
                gaussian_pyramid: A sequence of corresponding Gaussian pyramids to max depth - 1.
    """

    laplacian_pyramid = []
    gaussian_pyramid = construct_gaussian_pyramid(image)

    #   Subtract layers to max depth
    for i in range(len(gaussian_pyramid) - 1):
        laplacian_pyramid.append(gaussian_pyramid[i] - interpolate(gaussian_pyramid[i + 1]))
  
    #   Exclude last level of the gaussian pyramid as it has no correspondent in the laplacian pyramid
    return laplacian_pyramid, gaussian_pyramid[:-1]

def reconstruct_image(pyramid):
    """
        Reconstructs an image from a Laplacian pyramid.

        Parameters:
            pyramid: A Laplacian pyramid.

        Returns:
            reconstructed_image: A reconstructed image obtained by summing up the Laplacian pyramid.
    """

    #   Invert pyramid and start reconstruction from the smallest layer
    inverted_pyramid = pyramid[::-1]
    reconstructed_image = inverted_pyramid[0]
    
    #   Add up all the pyramid layers, reversing the process in construct_pyramids()
    for i in range(1, len(inverted_pyramid)):
        reconstructed_image = interpolate(reconstructed_image) + inverted_pyramid[i]
        
    return reconstructed_image

def blend_pyramids(A, B, M):
    """
        Blends two Laplacian pyramids using a Gaussian pyramid constructed from a mask as weights.
        Parameters:
                A: the first image
                B: the second iamge
                M: the mask
        Returns:
                blended_pyramid: a blended pyramid from two images and a mask.
    """
    #   Retrieve Laplacian and Gaussian pyramids of images A and B, as well as the mask, assuming that they are all of the same dimensions
    LA , _ = construct_pyramids(A)
    LB , _ = construct_pyramids(B)
    _ , GM = construct_pyramids(M)
    
    blended_pyramid = []
    
    #   Formula for blending pyramids that I lifted from https://github.com/twyunting/Laplacian-Pyramids.git and other correspondent sources
    for i in range(len(LA)):
        LS = GM[i] / 255 * LA[i] + (1 - GM[i] /255) * LB[i]
        blended_pyramid.append(LS)
        
    return blended_pyramid

def blend_image(img1, img2, mask):
    """
    Blends two images together based on a mask.

    Parameters:
        img1: The first image.
        img2: The second image.
        mask: A mask to use for weights in blending.

    Returns:
        blended_image: The blended image.
    """
    #   Split images into RGB channels, assuming all three are of the same dimensions
    img1R, img1G, img1B = cv2.split(img1)
    img2R, img2G, img2B = cv2.split(img2)
    mask, _, _ = cv2.split(mask)
    
    #   Apply pyramid blending to each channel
    R = reconstruct_image(blend_pyramids(img1R, img2R, mask))
    G = reconstruct_image(blend_pyramids(img1G, img2G, mask))
    B = reconstruct_image(blend_pyramids(img1B, img2B, mask))

    #   Merge the blended channels back into an image
    blended_image = cv2.merge((R, G, B))

    return blended_image