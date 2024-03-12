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
    be within bounds of the image. No check is performed to see
    if x_lower < x_upper or y_lower < y_upper.

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

@DeprecationWarning
def canny_edge(
    image: np.ndarray,
    gaussian_kernel_size: tuple[int, int] = (5, 5),
    canny_lower: int = 50,
    canny_upper: int = 150
) -> np.ndarray | np.ndarray:
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
    
    gray_image = image
    
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #   Apply Gaussian blur to reduce noise
    gray_image = cv2.GaussianBlur(gray_image, gaussian_kernel_size, 0)

    #   Perform Canny edge detection
    canny_edges = cv2.Canny(gray_image, canny_lower, canny_upper)

    return canny_edges, gray_image

@DeprecationWarning
def detect_contours_canny(
    image: np.ndarray,
    gaussian_kernel_size: tuple[int, int] = (5, 5),
    canny_lower: int = 50,
    canny_upper: int = 150
) -> np.ndarray:
    """
        Detects contours in an image using Canny edge detection and draws bounding boxes around the contours.

        Parameters:
            image (np.ndarray): The input image.
            canny_threshold1 (int): The first threshold for the Canny edge detector. Default is 50.
            canny_threshold2 (int): The second threshold for the Canny edge detector. Default is 150.

        Returns:
            np.ndarray: The image with bounding boxes drawn around detected contours.
    """
    
    #   Retrieve canny edges
    canny_edges, grey_image = canny_edge(image, gaussian_kernel_size, canny_lower, canny_upper)

    #   Find contours in the Canny-edged image
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contoured_image = grey_image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(contoured_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return contoured_image

@DeprecationWarning
def isolate_colorspace(
    image: np.ndarray,
    hsv_lower: list[int, int, int] = [0, 100, 100],
    hsv_upper: list[int, int, int] = [10, 255, 255],
    gaussian_kernel_size: tuple[int, int] = (5, 5),
    dilation_kernel_size: tuple[int, int] = (5, 5),
    erosion_kernel_size: tuple[int, int] = (5, 5),
    dilation_iterations: int = 20,
    erosion_iterations: int = 1,
) -> np.ndarray:
    
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
    if gaussian_kernel_size is not None: assert gaussian_kernel_size == (0, 0) or (gaussian_kernel_size[0] % 2 != 0 and gaussian_kernel_size[1] % 2 != 0), "Gaussian kernel dimensions must be odd."
    if dilation_kernel_size is not None: assert dilation_kernel_size == (0, 0) or (dilation_kernel_size[0] % 2 != 0 and dilation_kernel_size[1] % 2 != 0), "Dilation kernel dimensions must be odd."
    if erosion_kernel_size is not None: assert erosion_kernel_size == (0, 0) or (erosion_kernel_size[0] % 2 != 0 and erosion_kernel_size[1] % 2 != 0), "Erosion kernel dimensions must be odd."
    if dilation_iterations is not None: assert dilation_iterations >= 0, "Dilation iterations must be a non-negative integer."
    if erosion_iterations is not None: assert erosion_iterations >= 0, "Erosion iterations must be a non-negative integer."
    
    #   Convert BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #   Apply gaussian_blur if parameter supplied
    if gaussian_kernel_size != (0, 0):
        #   Apply Gaussian blur to reduce noise
        hsv_image = cv2.GaussianBlur(hsv_image, gaussian_kernel_size, 0)
    
    #   Threshold the HSV image to get only some colors
    mask = cv2.inRange(hsv_image, np.array(hsv_lower), np.array(hsv_upper))
    
    if dilation_iterations != 0 and dilation_kernel_size != (0, 0):
        #   Perform morphological dilation to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilation_kernel_size)
        mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)

    if erosion_iterations != 0 and erosion_kernel_size != (0, 0):
        #   Perform erosion to remove background 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, erosion_kernel_size)
        mask = cv2.erode(mask, kernel, iterations=erosion_iterations)

    #   Apply mask to the image
    isolated_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

    return isolated_image

def naive_threshold(
    image: np.ndarray,
    lower_threshold: int = 100,
    upper_threshold: int = 255
) -> np.ndarray:
    """
        This function applies naive thresholding to an image.

        Parameters:
            image (np.ndarray): The input image.
            lower_threshold (int): The lower threshold value.
            upper_threshold (int): The upper threshold value.

        Returns:
            np.ndarray: The thresholded image.
    """
        
    #   Apply naive thresholding to an image, setting pixels not within [lower_threshold, upper_threshold] to 0
    _, thresholded_image = cv2.threshold(image, lower_threshold, upper_threshold, cv2.THRESH_BINARY)

    return thresholded_image

def extract_black_regions(image: np.ndarray) -> np.ndarray:
    """
        Extracts black regions from an image and converts non-black regions to white.

        Parameters:
            image (numpy.ndarray): The input image in BGR format.

        Returns:
            numpy.ndarray: Image with black regions extracted and non-black regions converted to white.
    """
    #   Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #   Threshold the grayscale image to get black regions
    _, black_regions = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)
    
    #   Invert the black regions to get white regions for non-black areas
    white_regions = cv2.bitwise_not(black_regions)
    
    #   Convert the white regions to BGR format for visualization
    white_regions_bgr = cv2.cvtColor(white_regions, cv2.COLOR_GRAY2BGR)
    
    return white_regions_bgr

def count_white_pixels(image: np.ndarray) -> int:
    """
        Count the number of white pixels (255) in an image.

        Parameters:
            binary_image (np.ndarray): The binary image where white pixels are represented as 255.

        Returns:
            int: The number of white pixels in the binary image.
    """
    return np.count_nonzero(image == 255)

class LinearRegression:
    """
        A LinearRegression class that represents a simple linear regression model.

        This class provides methods to fit the model to data, make predictions, and retrieve the model parameters.

        Attributes:
            slope (float): The slope of the linear regression line.
            intercept (float): The y-intercept of the linear regression line.

        Methods:
            __init__(): Initializes the LinearRegression object with default slope and intercept values.
            __str__(): Returns a string representation of the linear regression equation.
            fit(data): Fits the linear regression model to the given data.
            predict(input): Predicts the output value based on the input using the linear regression equation.
            get_parameters(): Returns the slope and intercept of the linear regression model.
    """
    def __init__(self):
        self.slope = 0
        self.intercept = 0
    
    def __str__(self):
        return f"Y = {self.slope}(X) + {self.intercept}"
    
    def fit(self, data):
        x = data["PixelCount"].values
        y = data["Volume(in ml)"].values
        self.slope, self.intercept = np.polyfit(x, y, 1)

    #   y = mx + b, since we have observed a linear relationship between the pixel count and the expected volume of the red liquid
    def predict(self, input):
        return self.slope * input + self.intercept
        
    def get_parameters(self):
        return self.slope, self.intercept