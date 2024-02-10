"""
    Name: Computer Vision Laboratory 02
    Author: Rene Andre Bedonia Jocsing
    Date Modified: 02/10/2024 
    Usage: python lab02.py
    Description:
        This is a Python script that utilizes the cv2 package to implement an image filtering function
        and use it to create hybrid images using a simplified version of the SIGGRAPH 2006
        paper by Oliva, Torralba, and Schyns. Hybrid images are static images that change
        in interpretation as a function of the viewing distance. The basic idea is that high
        frequency tends to dominate perception when it is available, but, at a distance, only
        the low frequency (smooth) part of the signal can be seen. By blending the high frequency
        portion of one image with the low-frequency portion of another, you get a hybrid image 
        that leads to different interpretations at different distances.
        All the necessary image filtering functions can be found in hybrid.py.
        The testing suite can be found in lab02.py and should be used to access hybrid.py.
"""

from hybrid import cross_correlation, convolution, gaussian_blur

import cv2
import numpy as np

def main():
    test_convolution()

'''UTILITIES'''
#   Helper method to show an image
def show_image(name, image):
    #   Show image in a normal window
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''TESTING'''

def test_correlation():
    #   Assume that image.png already exists
    img = cv2.imread('input/image.png')
    
    show_image('Samurai Doge', img)
    
    #   Mean Filter Kernel
    #kernel = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)

    #   Horizontal Sobel Kernel for Edge Detection
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    #   Vertical Sobel Kernel for Edge Detection
    #kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
 
    # Apply cross-correlation
    result = cross_correlation(img, kernel)

    show_image('Samurai Doge', result)

def test_convolution():
    #   Assume that image.png already exists
    img = cv2.imread('input/image.png')  # Assuming grayscale image
    
    show_image('Samurai Doge', img)
    
    #   Mean Filter Kernel
    #kernel = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)

    #   Horizontal Sobel Kernel for Edge Detection
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    #   Vertical Sobel Kernel for Edge Detection
    #kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    # Apply convolution
    result = convolution(img, gaussian_blur(5, 3, 3))

    show_image('Samurai Doge', result)
    
    
if __name__ == "__main__":
    main()