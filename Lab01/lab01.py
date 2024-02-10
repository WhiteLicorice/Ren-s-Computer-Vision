"""
    Name: Computer Vision Laboratory 01
    Author: Rene Andre Bedonia Jocsing
    Date Modified: 02/04/2024 
    Usage: python lab01.py
    Description:
        This is a Python script that utilizes the cv2 package to replicate the process done in this video https://fb.watch/pWLNqOIQPE/ from Facebook.
        The script outputs .jpg files to illustrate each stage of the process.
        An input image named 'image.png' in the /input directory is required.
"""

"""PACKAGES"""

import cv2
import numpy as np

"""PREAMBLE"""

#   This square image has dimensions 2000 x 2000 pixels (use 'nice' dimensions to avoid complicating things)
IMAGE_PATH = "./input/image.png"
OUTPUT_DIRECTORY = "./output"
NUMBER_OF_STRIPS = 20
WINDOW_NAME = "I don't want to be horny anymore... I just want to be happy."

"""SCRIPT"""

def main():
    #   Parse image from IMAGE_PATH (assume image already exists and has 'nice' dimensions)
    image = cv2.imread(IMAGE_PATH)
    
    #   Show original image
    show_image(WINDOW_NAME, image)
    
    #   Divide image into vertical strips
    vertical_strips = divide_vertical(image, NUMBER_OF_STRIPS)
    save_strips(vertical_strips, "vertical")

    #   Take alternating vertical strips
    image_odd, image_even = alternate_merge_strips_vertical(vertical_strips)
    cv2.imwrite(f'{OUTPUT_DIRECTORY}/odd_vertical.jpg', image_odd)
    cv2.imwrite(f'{OUTPUT_DIRECTORY}/even_vertical.jpg', image_even)
    
    #   Show combined alternating vertical strips and save
    #   If axis = 1, then horizontal axis (place images side by side), if axis = 0, then vertical axis (mount images on top of each other)
    combined_image_vertical = np.concatenate((image_odd, image_even), axis = 1)
    cv2.imwrite(f'{OUTPUT_DIRECTORY}/combined_vertical.jpg', combined_image_vertical)
    show_image(WINDOW_NAME, combined_image_vertical)
    
    #   Divide image from previous step into horizontal strips
    horizontal_strips = divide_horizontal(combined_image_vertical, NUMBER_OF_STRIPS)
    save_strips(horizontal_strips, "horizontal")

    #   Take alternating horizontal strips
    image_odd, image_even = alternate_merge_strips_horizontal(horizontal_strips)
    cv2.imwrite(f'{OUTPUT_DIRECTORY}/odd_horizontal.jpg', image_odd)
    cv2.imwrite(f'{OUTPUT_DIRECTORY}/even_horizontal.jpg', image_even)
    
    #   Show combined alternating horizontal strips and save
    combined_image_horizontal = np.concatenate((image_odd, image_even), axis = 1)       
    cv2.imwrite(f'{OUTPUT_DIRECTORY}/combined_horizontal.jpg', combined_image_horizontal)
    show_image(WINDOW_NAME, combined_image_horizontal)
    
"""UTILITIES"""

#   Helper method for saving strips (key is used as a filename identifier)
def save_strips(strips, key):
    #   Save and/or display each strip
    for i, strip in enumerate(strips):
        cv2.imwrite(f'{OUTPUT_DIRECTORY}/strip_{key}_{i + 1}.jpg', strip)  #    Save each strip to a uniquely named file
    
#   Helper method for dividing an image into N number of even vertical strips
def divide_vertical(image, num_strips = 10):
    #   Get image dimensions
    _, width = image.shape[:2]
    
    #   Calculate the width of each strip
    strip_width = width // num_strips

    #   Image data structure is in the form: image[height[...], width[...]]
    vertical_strips = [image[ : , (i * strip_width) : ((i + 1) * strip_width) ] for i in range(num_strips)]

    return vertical_strips

#   Helper method for dividing an image into N number of even horizontal strips
def divide_horizontal(image, num_strips = 10):
    #   Get image dimensions
    height, _ = image.shape[:2]

    #   Calculate the width of each strip
    strip_height = height // num_strips

    #   Image data structure is in the form: image[height[...], width[...]]
    horizontal_strips = [image[(i * strip_height) : ((i + 1) * strip_height) , : ] for i in range(num_strips)]

    return horizontal_strips

#   Helper method to merge strips by placing them side by side
def alternate_merge_strips_vertical(strips):
    #   Call helper method to get alternating strips
    odd_strips, even_strips = get_alternating_strips(strips)
    
    #   Concatenate odd and even strips, placing them side by side
    image_odd = np.concatenate(odd_strips, axis = 1)
    image_even = np.concatenate(even_strips, axis = 1)
    
    return image_odd, image_even

#   Helper method to merge strips by mounting them on top of each other
def alternate_merge_strips_horizontal(strips):
    #   Call helper method to get alternating strips
    odd_strips, even_strips = get_alternating_strips(strips)
    
    #   Concatenate odd and even strips, placing them on top of each other
    image_odd = np.concatenate(odd_strips, axis = 0)
    image_even = np.concatenate(even_strips, axis = 0)
    
    return image_odd, image_even

#   Helper method to get alternating odd-even strips from an array of strips
def get_alternating_strips(strips):
    #   Separate strips into odd and even, alternating pattern
    odd_strips = [strip for i, strip in enumerate(strips) if i % 2 != 0]
    even_strips = [strip for i, strip in enumerate(strips) if i % 2 == 0]
    
    return odd_strips, even_strips

#   Helper method to show an image
def show_image(name, image):
    #   Show image in a normal window
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""MODULE DECLARATION"""

if __name__ == "__main__":
    #print(f"v{cv2.__version__} of OpenCV installed!")
    #print(f"v{np.__version__} of NumPy installed!")
    main()