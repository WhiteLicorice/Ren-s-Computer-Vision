import cv2
import numpy as np

def crop(image: np.ndarray, 
         x_lower: int = None, 
         x_upper: int = None, 
         y_lower: int = None, 
         y_upper: int = None) -> np.ndarray:
    
    """
        Crops an image along the x and y axis. The x and y values must
        be within bounds of the image.

        Parameters:
                image: the original image
                x_lower: the lower bound of x (default: 0)
                x_upper: the upper bound of x (default: image.width)
                y_lower: the lower bound of y (default: 0)
                y_upper: the upper bound of y (default: image.height)
                
        Returns:
                cropped_image: the image cropped according to the window formed by x_lower, x_upper, y_lower, and y_upper
    """
    
    assert y_lower is None or y_lower >= 0 and y_upper is None or y_upper <= image.shape[0], "y out of bounds"
    assert x_lower is None or x_lower >= 0 and x_upper is None or x_upper <= image.shape[1], "x out of bounds"
    
    #   Crop the image according to the parameters
    cropped_image = image[y_lower or 0 : y_upper or image.shape[0], x_lower or 0: x_upper or image.shape[1]]
    
    return cropped_image

#   TODO: Add other approaches for detecting the fluid threshold (eg. sobel edge detection, surf, etc.)
#   TODO: Anything goes!

def canny_edge(image: np.ndarray,
               gaussian_kernel_size: tuple[int, int] = (5, 5),
               canny_lower: int = 50,
               canny_upper: int = 225):
    """
        Performs canny edge detection on an image. Automatically converts to greyscale
        and applies gaussian blur to reduce noise.

        Parameters:
                image: the original image
                gaussian_kernel_size: the size of the guassian kernel to be applied when blurring.
                canny_lower: the lower threshold of the canny edge detection filter.
                canny_upper: the upper threshold of the canny edge detection filter.
                
        Returns:
                canny_edges: the high-pass filtered image with canny edges revealed.
    """
    assert gaussian_kernel_size[0] % 2 != 0 and gaussian_kernel_size[1] % 2 != 0, "Gaussian kernel dimensions must be odd."
    
    #   Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #   Apply Gaussian blur to reduce noise
    gray_image = cv2.GaussianBlur(gray_image, gaussian_kernel_size, 0)

    #   Perform Canny edge detection
    canny_edges = cv2.Canny(gray_image, canny_lower, canny_upper)

    return canny_edges