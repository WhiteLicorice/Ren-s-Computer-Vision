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
import glob

from thresholding import crop, naive_threshold, extract_black_regions, count_white_pixels

def main():
    sample_images = {
        '50ml': r'input/50ml/1.jpg',
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
    
    #   Collect images from all the directories
    _50ml_images = collect_images('50ml')
    _100ml_images = collect_images('100ml')
    _150ml_images = collect_images('150ml')
    _200ml_images = collect_images('200ml')
    _250ml_images = collect_images('250ml')
    _300ml_images = collect_images('300ml')
    _350ml_images = collect_images('350ml')
    _A_images = collect_images('A')
    _B_images = collect_images('B')
    _C_images = collect_images('C')

    all_images = { }
    
    #   Collate images into dictionary
    all_images.update(_50ml_images)
    all_images.update(_100ml_images)
    all_images.update(_150ml_images)
    all_images.update(_200ml_images)
    all_images.update(_250ml_images)
    all_images.update(_300ml_images)
    all_images.update(_350ml_images)
    all_images.update(_A_images)
    all_images.update(_B_images)
    all_images.update(_C_images)

    #   Compute pixel values of the fluid in each image via naive thresholding
    for directory, image_paths in all_images.items():
        for image_path in image_paths:
            #   Pipeline: crop image as close to the bottle as possible -> threshold image to black out the fluid -> invert the threshold to have fluid as white regions
            cropped_image = cv2.imread(image_path)
            cropped_image = crop(cropped_image, x_lower=1500, x_upper=2000, y_lower=550, y_upper=1500)
            cropped_image = naive_threshold(cropped_image, 100, 225)
            cropped_image = extract_black_regions(cropped_image)
            
            print(f"{cropped_image.shape}: {count_white_pixels(cropped_image)} px -> {directory}")
            #show_image('Image', cropped_image)
            #break
        
#   Helper method to show an image
def show_image(image_label, image):
    #   Show image in a normal window
    cv2.namedWindow(image_label, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_label, image.shape[1], image.shape[0])
    cv2.imshow(image_label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#   Helper method for collecting all the images in a directory  
def collect_images(input_directory):
    search_pattern = f"input/{input_directory}/*.jpg"
    image_files = glob.glob(search_pattern)
    return {input_directory: image_files}

if __name__ == "__main__":
    main()