"""
    Name: Computer Vision Laboratory 04
    Author(s): Rene Andre Bedonia Jocsing & Ron Gerlan Naragdao
    Date Modified: 03/08/2024 
    Usage: python lab04.py
    Description:
        TODO: Place activity description here.
"""

import cv2
import numpy as np

from thresholding import crop, canny_edge, isolate_colorspace, remove_colorspace

#   TODO: Find a way to dynamically threshold an image: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
def main():
    sample_images = {
        '50ml': r'input/50mL/1.jpg',
        '100ml': r'input/100ml/1.jpg',
        '150ml': r'input/150ml/1.jpg',
        '200ml': r'input/200ml/1.jpg',
        '250ml': r'input/250ml/1.jpg',
        '300ml': r'input/300ml/1.jpg',
        '350ml': r'input/350ml/1.jpg',
        'A': r'input/A/1.jpg',
        'B': r'input/B/1.jpg',
        'C': r'input/C/1.jpg',
    }
    
    #   TODO: Calibrate numbers when detecting edges (this might be different for each ML category)
    #   Preliminary investigation of images
    for label, path in sample_images.items():
        image = cv2.imread(path)
        cropped_image = crop(image, x_lower=0, x_upper=2300, y_lower=500, y_upper=image.shape[0])
        cropped_image = isolate_colorspace(cropped_image, [0, 10, 10], [10, 255, 255], (51, 51), (15, 15), 20, 10)
        #cropped_image = canny_edge(cropped_image)
        
        print(cropped_image.shape)
        show_image(label, cropped_image) 
        #break

    #   TODO: Find contours based on edges
    #   TODO: Fit a bounding box around the contours
    #   TODO: Find the area of the bounding box
    
#   Helper method to show an image
def show_image(image_label, image):
    #   Show image in a normal window
    cv2.namedWindow(image_label, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_label, image.shape[1], image.shape[0])
    cv2.imshow(image_label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()