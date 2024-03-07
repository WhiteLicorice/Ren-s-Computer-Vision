import cv2
import numpy as np
from scipy.signal import convolve2d as convolve

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
        #   Blur by quadrupling the kernel, since we interpolated by a factor of 2(width) and 2(height)
        interpolated_channel = convolve(interpolated_channel, 4 * kernel, mode='same')
        
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
        decimated_channel = convolve(channel, kernel, mode='same')[::2, ::2]
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
    
    #   Let the last layer of the laplacian pyramid be the last layer of its gaussian pyramid
    laplacian_pyramid.append(gaussian_pyramid[-1])
    
    #   Slight but harmless optimization -> return the gaussian pyramid along with the laplacian pyramid
    return laplacian_pyramid, gaussian_pyramid

def reconstruct_image(pyramid):     #   Coolest thing I learned in this lab activity
    """
        Reconstructs an image from a Laplacian pyramid.

        Parameters:
            pyramid: A Laplacian pyramid.

        Returns:
            reconstructed_image: A reconstructed image obtained by summing up the Laplacian pyramid.
    """

    #   Grab the last layer in the laplacian pyramid for reverse traversal
    reconstructed_image = pyramid[-1]
    
    #   Add up all the pyramid layers from the smallest layer up to the largest layer, reversing the process in construct_pyramids()
    #   In reality, this is the same as: for i in range(1, invert(pyramid).length) but does not require an explicit invert() function
    for i in range(len(pyramid) - 2, -1, -1):
        reconstructed_image = interpolate(reconstructed_image) + pyramid[i]
        
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
    #   Retrieve laplacian pyramids of the two images and the gaussian pyramid of the mask
    LA , _ = construct_pyramids(A)
    LB , _ = construct_pyramids(B)
    GR = construct_gaussian_pyramid(M)
    
    blended_pyramid = [ ]
    
    #   Formula for blending pyramids that I lifted from the blackboard (amazing!)
    for i in range(len(GR)):
        LS = GR[i] / 255 * LA[i] + (1 - GR[i] /255) * LB[i]
        blended_pyramid.append((LS))    #   bound() here sort of blends the image colors as well! ~ found out after 3 weeks of experimentation (manifesting moved deadlines)
        
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
        No error-handling occurs if the images are not of the same size, but an error is thrown.

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