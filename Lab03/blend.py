import cv2
import numpy as np

def interpolate(img):
    """
        Interpolates an image with upsampling rate r = 2.

        Return: the interpolated image with upsampling rate r = 2.
    """
    
    img = img.astype(np.float64)    #   Make image precise to mitigate detail loss when interpolating/decimating
    upsampling_rate = 2.0           #   Set r = 2.0
    height, width = img.shape[:2]   #   Grab height and width of image

    #   Upscale height and width by a factor of r = 2.0
    new_height = int(height * upsampling_rate)
    new_width = int(width * upsampling_rate)

    #   Interpolate via cv2 linear interpolation, worse than INTER_BICUBIC but faster and still looks okay on small images
    interpolated_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return interpolated_image

def decimate(img):
    """
        Decimates an image with downsampling rate r = 2.

        Return: the decimated image with downsampling rate r = 2.
    """
    
    img = img.astype(np.float64)    #   Make image precise to mitigate detail loss when interpolating/decimating
    downsampling_rate = 2.0         #   Set r = 2.0
    height, width = img.shape[:2]   #   Grab height and width of image

    #   Downsample height and width by a factor of r = 2.0
    new_height = int(height / downsampling_rate)
    new_width = int(width / downsampling_rate)

    #   Decimate via cv2 inter-area interpolation, which is useful for downsampling according to StackOverFlow gurus
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

    for i in range(len(gaussian_pyramid) - 1):
        #DEPRECATED: Runs into shape error when pyramid layers have odd dimensions due to imprecise floating-point arithmetic
        # print(f"{gaussian_pyramid[i].shape} - {interpolate(gaussian_pyramid[i + 1]).shape}")
        # laplacian_pyramid.append(gaussian_pyramid[i] - interpolate(gaussian_pyramid[i + 1]))
        
        current_level = gaussian_pyramid[i]
        next_level = interpolate(gaussian_pyramid[i + 1])
        
        #   Account for miniature pixel errors due to how interpolation works
        if current_level.shape != next_level.shape:
            current_level = current_level[: next_level.shape[0], :next_level.shape[1]]
            gaussian_pyramid[i] = current_level     #   Save adjustments
        
        #print(f"{current_level.shape} - {next_level.shape}")
        laplacian_pyramid.append(current_level - next_level)
        
    #   Exclude last level of the gaussian pyramid as it has no correspondent in the laplacian pyramid
    return laplacian_pyramid, gaussian_pyramid[:-1]

def reconstruct_image(pyramid):
    """
        Reconstructs an image from a Laplacian pyramid.

        Parameters:
            pyramid: A Laplacian pyramid.

        Returns:
            stack: A reconstructed image obtained by summing up the Laplacian pyramid.
    """

    #   Start with the smallest pyramid so we need to reverse the order
    inverted_pyramid = pyramid[::-1]
    
    #   Start with the first level of the inverted pyramid
    reconstructed_image = inverted_pyramid[0]
    
    #   Loop through the levels of the pyramid starting from the second level
    for i in range(1, len(inverted_pyramid)):
        next_level = inverted_pyramid[i]
        current_level = interpolate(reconstructed_image)
        
        #   Account for minute pixel differences by resizing the current reconstructed image to match the next level
        if next_level.shape != current_level.shape:
            current_level = cv2.resize(reconstructed_image, (inverted_pyramid[i].shape[1], inverted_pyramid[i].shape[0]), interpolation=cv2.INTER_LINEAR)

        #   Add the details of the next level in the pyramid to the reconstructed image
        reconstructed_image = current_level + next_level
        
        #   Remove some artifacts by clipping the values outside the valid range
        reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
        
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
    LA, _ = construct_pyramids(A)
    LB, _ = construct_pyramids(B)
    _, GM = construct_pyramids(M)
  
    blended_pyramid = []
    
    #   Formula for blending pyramids that I lifted from https://github.com/twyunting/Laplacian-Pyramids.git
    for i in range(len(LA)):
        LS = GM[i] / 255 * LA[i] + (1 - GM[i] / 255) * LB[i]
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
