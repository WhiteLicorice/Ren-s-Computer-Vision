import cv2
import numpy as np
import scipy as sp

def interpolate(img):
    """
        Interpolates an image with upsampling rate r=2.

        Return: the interpolated image with upsampling rate r=2.
    """
    
    upsampling_rate = 2.0           #   Set r = 2.0
    height, width = img.shape[:2]   #   Grab height and width of image

    #   Upscale height and width by a factor of r = 2.0
    new_height = int(height * upsampling_rate)
    new_width = int(width * upsampling_rate)

    #   Interpolate via cv2 bicubic interpolation
    interpolated_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return interpolated_image

def decimate(img):
    """
        Decimates an image with downsampling rate r=2.

        Return: the decimated image with downsampling rate r=2.
    """

    downsampling_rate = 2.0         #   Set r = 2.0
    height, width = img.shape[:2]   #   Grab height and width of image

    #   Downsample height and width by a factor of r = 2.0
    new_height = int(height // downsampling_rate)
    new_width = int(width // downsampling_rate)

    #   Decimate via cv2 inter-area interpolation, which is used specifically for downsampling
    decimated_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return decimated_image

def construct_gaussian_pyramid(image):
    """
        Constructs a Gaussian pyramid to max depth.
        Parameters:
                image: the original image (i.e. base of the pyramid)
        Returns:
                gaussian_pyramid: A sequence of Gaussian pyramids to max depth.
    """

    gaussian_pyramid = [image, ]

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

    for i in range(len(gaussian_pyramid) - 1):
        #DEPRECATED: Runs into shape error when pyramids have odd dimensions
        # print(f"{gaussian_pyramid[i].shape} - {interpolate(gaussian_pyramid[i + 1]).shape}")
        # laplacian_pyramid.append(gaussian_pyramid[i] - interpolate(gaussian_pyramid[i + 1]))
        
        current_level = gaussian_pyramid[i]
        next_level = interpolate(gaussian_pyramid[i + 1])
        
        if current_level.shape != next_level.shape:
            current_level = current_level[: next_level.shape[0], :next_level.shape[1]]
            gaussian_pyramid[i] = current_level     #   Save adjustments
        
        #print(f"{current_level.shape} - {next_level.shape}")
        laplacian_pyramid.append(current_level - next_level)
        
    #   Exclude last level of the gaussian pyramid as it has no correspondent in the laplacian pyramid
    return laplacian_pyramid, gaussian_pyramid[:-1]

#   TODO: Fix pyramid blending. Sometimes, GM[i] has a third axis while LA[i] has only two.
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
    LA, _ = construct_pyramids(A)
    LB, _ = construct_pyramids(B)
    _, GM = construct_pyramids(M)
  
    blended_pyramid = []
    for i in range(len(LA)):
        LS = GM[i] / 255 * LA[i] + (1 - GM[i] / 255) * LB[i]
        blended_pyramid.append(LS)
  
    return blended_pyramid

#   TODO: Fix pyramid reconstruction. No handling of pixel inaccuracies with cv2.resize() similar to the one in construct_pyramids().
def reconstruct_image(pyramid):
    """
        Reconstructs an image from a Laplacian pyramid.

        Parameters:
            pyramid: A list of Laplacian pyramid levels.

        Returns:
            stack: A reconstructed image obtained by collapsing the Laplacian pyramid.
    """
    
    #rows, cols = pyramid[0].shape[:2]
    #res = np.zeros((rows, cols + cols // 2), dtype=np.double)
    
    #   Start with the smallest pyramid so we need to reverse the order
    revPyramid = pyramid[::-1]
    stack = revPyramid[0]
    
    #   Start with the second index
    for i in range(1, len(revPyramid)):
        stack = interpolate(stack) + revPyramid[i]  #   Upsampling simultaneously
        
    return stack

#   TODO: Fix image blending. This relies on reconstruct_image() and blend_pyramids() which are quite buggy at the moment.
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
    # Split images into Red, Green, and Blue channels
    img1R, img1G, img1B = cv2.split(img1)
    img2R, img2G, img2B = cv2.split(img2)
    mask, _, _ = cv2.split(mask)

    # Apply pyramid blending to each channel
    R = reconstruct_image(blend_pyramids(img1R, img2R, mask))
    G = reconstruct_image(blend_pyramids(img1G, img2G, mask))
    B = reconstruct_image(blend_pyramids(img1B, img2B, mask))

    # Merge the channels back into an image
    blended_image = cv2.merge((R, G, B))

    return blended_image
