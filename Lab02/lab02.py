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
    img = cv2.imread('input/image.png')  # Assuming grayscale image
    
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