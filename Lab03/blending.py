import cv2
import numpy as np
from scipy.ndimage import convolve

#   Define a binomial 5-tap binomial filter from https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
#   According to the docs, this is the kernel that OpenCV pyrDown and pyrUp uses
kernel = (1.0/256) * np.array([
                                [1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]
                                                    ])

def interpolate(img):
    """
        Interpolates an image with upsampling rate r = 2.

        Parameters:
                image: the original image 
        Returns:
                interpolated_image: the image interpolated by a factor of 2.
    """
    
    channels = split_rgb(img)
    processed_channels = [ ]
    
    for channel in channels:
        #   Create a blank channel that is twice the area of the original image
        interpolated_channel = np.zeros((2 * channel.shape[0], 2 * channel.shape[1]))
        #   Upsample each channel, by taking pixels from the original channel and using them to fill every other pixel of the blank channel
        interpolated_channel[::2, ::2] = channel
        #   Blur by quadrupling the kernel, since we interpolated by a factor of 2
        interpolated_channel = convolve(interpolated_channel, 4 * kernel, mode='constant')
        processed_channels.append(interpolated_channel)
        
    interpolated_image = cv2.merge(processed_channels)
    
    return interpolated_image

def decimate(img):
    """
        Decimates an image with downsampling rate r = 2.
        
        Parameters:
                image: the original image 
        Returns:
                interpolated_image: the image decimated by a factor of 2.

    """
    
    channels = split_rgb(img)
    processed_channels = [ ]
    
    for channel in channels:
        #   Apply Gaussian blur then downsample each channel by taking every other pixel
        decimated_channel = convolve(channel, kernel, mode='constant')[::2, ::2]
        processed_channels.append(decimated_channel)
        
    decimated_image = cv2.merge(processed_channels)
    
    return decimated_image

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

    #   Subtract layers to max depth, laplacian_i = gaussian_i - interpolate(gaussian_i+1)
    for i in range(len(gaussian_pyramid) - 1):
        laplacian_pyramid.append(gaussian_pyramid[i] - interpolate(gaussian_pyramid[i + 1]))
  
    #   Discard the last level of the gaussian pyramid as it has no correspondent in the laplacian pyramid
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
        
    return bound(reconstructed_image)

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
    #   Retrieve Laplacian and Gaussian pyramids of images A and B, as well as the gaussian mask, assuming that they are all of the same dimensions
    LA , _ = construct_pyramids(A)
    LB , _ = construct_pyramids(B)
    _ , GM = construct_pyramids(M)
    
    blended_pyramid = [ ]
    
    #   Formula for blending pyramids that I lifted from the blackboard (amazing!)
    for i in range(len(LA)):
        LS = GM[i] / 255 * LA[i] + (1 - GM[i] /255) * LB[i]
        blended_pyramid.append(bound(LS))
        
    return blended_pyramid

def split_rgb(img):
    """
        Wrapper around cv2.split() for splitting an image into individual channels.
        Parameters:
            img: the image to be splitted.
        Returns:
            channels: an array-like of the splitted channels.
    """
    return cv2.split(img)

def bound(img):
    """
        Clip values between (0, 255) per pixel in the image. This is needed because there are instances where values are out of bounds.
        Out of bound values in cv2 images generate noise and are unpleasant to look at.
        Parameters:
            img: the image to be normalized.
        Returns:
            clipped_image: the image with pixel values normalized to (0, 255) inclusive.
    """
    
    #   Replace all pixel values less than 0 with 0, and values greater than 255 with 255, then return as an array of numpy integers
    return np.where(img < 0, 0, np.where(img > 255, 255, img)).astype(np.uint8)

def blend_image(img1, img2, mask):
    """
    Blends two images together based on a mask. All three images are assumed to be of the same size.
    No check is performed to see if this assumption is met.

    Parameters:
        img1: The first image.
        img2: The second image.
        mask: A mask to use for weights in blending.

    Returns:
        blended_image: The blended image.
    """
    
    assert img1.shape[0] == img2.shape[0] and img1.shape[1] == img2.shape[1], "Images are not of equal dimensions."
    assert img1.shape[0] == mask.shape[0], "Mask does not match dimensions of the images."
    
    #   Chain the channel-agnostic functions above to generate a multi-resolution blended image? Not sure if that's what it's called.
    return reconstruct_image(blend_pyramids(img1, img2, mask))