"""
    Name: Computer Vision Laboratory 04
    Author: Rene Andre Bedonia Jocsing
    Date Modified: 03/07/2024 
    Usage: python lab04.py
    Description:
        TODO: Place description here.
"""

import cv2
import numpy as np

#   TODO: Find a way to dynamically threshold an image: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
def main():
    image = cv2.imread("input/absolute500ml.png", cv2.IMREAD_GRAYSCALE)
    
    show_image("Bottle", image)
    
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    
    show_image("Thresholded", thresh)
    
    return

#   Helper method to show an image
def show_image(name, image):
    #   Show image in a normal window
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()