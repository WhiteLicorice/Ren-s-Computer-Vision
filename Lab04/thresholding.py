import cv2
import numpy as np

def crop(
    image: np.ndarray, 
    x_lower: int = None, 
    x_upper: int = None, 
    y_lower: int = None, 
    y_upper: int = None
) -> np.ndarray:
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
    
    # Crop the image according to the parameters
    cropped_image = image[y_lower or 0 : y_upper or image.shape[0], x_lower or 0: x_upper or image.shape[1]]
    
    return cropped_image

def canny_edge(
    image: np.ndarray,
    gaussian_kernel_size: tuple[int, int] = (5, 5),
    canny_lower: int = 50,
    canny_upper: int = 225
) -> np.ndarray:
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
    if gaussian_kernel_size is not None: assert gaussian_kernel_size[0] % 2 != 0 and gaussian_kernel_size[1] % 2 != 0, "Gaussian kernel dimensions must be odd."
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_image = cv2.GaussianBlur(gray_image, gaussian_kernel_size, 0)

    # Perform Canny edge detection
    canny_edges = cv2.Canny(gray_image, canny_lower, canny_upper)

    return canny_edges

import cv2
import numpy as np

def isolate_colorspace(
    image: np.ndarray,
    hsv_lower: list[int, int, int] = [0, 100, 100],
    hsv_upper: list[int, int, int] = [10, 255, 255],
    gaussian_kernel_size: tuple[int, int] = (5, 5),
    morph_kernel_size: tuple[int, int] = (5, 5),
    dilation_iterations: int = 20,
    erosion_iterations: int = 1,
):
    """
        This function isolates a specific color space in an image using HSV thresholding and morphological operations.

        Parameters:
            image (np.ndarray): The input image in BGR format.
            hsv_lower (list[int, int, int]): The lower threshold values for the HSV color space. Default is [0, 100, 100].
            hsv_upper (list[int, int, int]): The upper threshold values for the HSV color space. Default is [10, 255, 255].
            gaussian_kernel_size (tuple[int, int]): The size of the Gaussian kernel for blurring the image. Default is (5, 5).
            morph_kernel_size (tuple[int, int]): The size of the morphological kernel for dilation and erosion operations. Default is (5, 5).
            dilation_iterations (int): The number of iterations for the dilation operation. Default is 20.
            erosion_iterations (int): The number of iterations for the erosion operation. Default is 1.

        Returns:
            np.ndarray: The isolated color space image.

        Raises:
            AssertionError: If the image is in grayscale or if the input parameters are invalid.
    """
    
    assert len(image.shape) > 1, "Image must not be in grayscale."
    if hsv_lower is not None: assert len(hsv_lower) == 3, "Tuple must be in HSV format."
    if hsv_upper is not None: assert len(hsv_upper) == 3, "Tuple must be in HSV format."
    if hsv_lower is not None: assert hsv_lower[0] in range(0, 179 + 1) and hsv_lower[1] in range(0, 255 + 1) and hsv_lower[2] in range(0, 255 + 1), "HSV values out of range."
    if hsv_upper is not None: assert hsv_upper[0] in range(0, 179 + 1) and hsv_upper[1] in range(0, 255 + 1) and hsv_upper[2] in range(0, 255 + 1), "HSV values out of range."
    if gaussian_kernel_size is not None: assert gaussian_kernel_size[0] % 2 != 0 and gaussian_kernel_size[1] % 2 != 0, "Gaussian kernel dimensions must be odd."
    if morph_kernel_size is not None: assert morph_kernel_size[0] % 2 != 0 and morph_kernel_size[1] % 2 != 0, "Morphological kernel dimensions must be odd."
    if dilation_iterations is not None: assert dilation_iterations >= 0, "Dilation iterations must be a non-negative integer."
    if erosion_iterations is not None: assert erosion_iterations >= 0, "Erosion iterations must be a non-negative integer."
    
    #   Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #   Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, gaussian_kernel_size, 0)
    
    #   Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    
    if dilation_iterations != 0:
        # Perform morphological dilation to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
        mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

    if erosion_iterations != 0:
        # Perform erosion to remove background 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
        mask = cv2.erode(mask, kernel, iterations=erosion_iterations)

    # Apply mask to the blurred image
    isolated_image = cv2.bitwise_and(blurred_image, blurred_image, mask=mask)

    return isolated_image


def remove_colorspace(
    image: np.ndarray,
    hsv_lower: list[int, int, int] = [40, 40, 40],
    hsv_upper: list[int, int, int] = [70, 255, 255]
):
    assert len(image.shape) > 1, "Image must not be in grayscale."
    if hsv_lower is not None: assert len(hsv_lower) == 3, "Tuple must be in HSV format."
    if hsv_upper is not None: assert len(hsv_upper) == 3, "Tuple must be in HSV format."
    if hsv_lower is not None: assert hsv_lower[0] in range(0, 179 + 1) and hsv_lower[1] in range(0, 255 + 1) and hsv_lower[2] in range(0, 255 + 1), "HSV values out of range."
    if hsv_upper is not None: assert hsv_upper[0] in range(0, 179 + 1) and hsv_upper[1] in range(0, 255 + 1) and hsv_upper[2] in range(0, 255 + 1), "HSV values out of range."

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of color to be removed in HSV
    lower_bound = np.array(hsv_lower)
    upper_bound = np.array(hsv_upper)

    # Create a mask for the regions to be removed
    removal_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Invert the mask to keep the non-removed regions
    removal_mask_inv = cv2.bitwise_not(removal_mask)

    # Apply the mask to remove the specified color range from the original image
    image_without_color = cv2.bitwise_and(image, image, mask=removal_mask_inv)

    return image_without_color
