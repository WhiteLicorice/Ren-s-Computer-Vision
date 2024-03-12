"""
    Name: Computer Vision Laboratory 04
    Author(s): Rene Andre Bedonia Jocsing & Ron Gerlan Naragdao
    Date Modified: 03/08/2024 
    Usage: python lab04.py [IS_VERBOSE]
    Description:
        The goal of this laboratory exercise is to estimate the amount of liquid contained in a bottle.
        The directory 'guess' contains images of the bottle with unknown amounts of liquid. You are to guess these amounts.
        OpenCV image filtering, thresholding, or morphology operations are allowed. The script can be accessed
        in lab04.py and should be used to access the API in thresholding.py.
"""

import cv2
import numpy as np
import glob
import pandas as pd
import sys

from thresholding import crop, naive_threshold, extract_black_regions, count_white_pixels, LinearRegression

global IS_VERBOSE
IS_VERBOSE = False  #   Enable IS_VERBOSE flag for debugging and testing

def main():  
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

    #   Collate known images into dictionary
    all_images = { }
    all_images.update(_50ml_images)
    all_images.update(_100ml_images)
    all_images.update(_150ml_images)
    all_images.update(_200ml_images)
    all_images.update(_250ml_images)
    all_images.update(_300ml_images)
    all_images.update(_350ml_images)
    
    column_names = ['Volume(in ml)', 'PixelCount']
    data = pd.DataFrame(columns=column_names)

    #   Compute pixel values of the fluid in each image via naive thresholding
    for directory, image_paths in all_images.items():
        for image_path in image_paths:
            #   Extract the pixel count of the fluid region via pipelined steps
            fluid_pixel_count = extraction_pipeline(image_path)
            #   Prep the directory name for use as values in the volume column of the dataframe
            directory_int = int(directory[:-2])
            #   Store Volume-Pixel as a row in the dataframe
            data.loc[len(data)] = [directory_int, fluid_pixel_count]

    #   Fit a Linear Regression model on the data and print details
    model = LinearRegression()
    model.fit(data)
    slope, intercept = model.get_parameters()
    print(f"{Color.YELLOW}{model}{Color.END}")
    print(f"{Color.YELLOW}Slope: {slope}\nIntercept: {intercept}{Color.YELLOW}")
    
    #   Collate unknown images into dictionary
    unknown_images = { }
    unknown_images.update(_A_images)
    unknown_images.update(_B_images)
    unknown_images.update(_C_images)
    
    #   Run pipeline on collated images and get predictions and mean prediction for each directory
    if IS_VERBOSE: print("Predictions in Mililiters")
    for directory, image_paths in unknown_images.items():
        if IS_VERBOSE: print(f"{Color.YELLOW}Directory {directory}{Color.END}")
        predictions = [ ]
        for image_path in image_paths:
            img_white_pixel_count = extraction_pipeline(image_path)
            prediction = model.predict(img_white_pixel_count)
            predictions.append(prediction)
            if IS_VERBOSE: print(f"{prediction}")
        print(f"{Color.GREEN}Directory {directory} Mean Volume Prediction in Mililiters: {np.mean(predictions)}{Color.END}")
            
"""TESTING SUITE"""
#   Test effect of crop + naive_threshold on images... Good enough for our purposes!
def test_naivethresholding():
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
    
    #   Pipeline: crop image as close to the bottle as possible -> threshold image to black out the fluid -> invert the threshold to have fluid as white regions
    for image_path in sample_images:
        cropped_image = cv2.imread(image_path)
        cropped_image = crop(cropped_image, x_lower=1500, x_upper=2000, y_lower=550, y_upper=1500)
        cropped_image = naive_threshold(cropped_image, 100, 225)
        cropped_image = extract_black_regions(cropped_image)
        if IS_VERBOSE: print(f"{cropped_image.shape} : {count_white_pixels(cropped_image)}")
        show_image(f"{image_path}", image_path)

<<<<<<< HEAD
"""UTILITIES"""
#   Helper method to extract pixel count of the red fluid region in the images         
def extraction_pipeline(image_path):
=======
    all_images = { }
    unknown_images = { }
    
    #   Collate images into dictionary
    all_images.update(_50ml_images)
    all_images.update(_100ml_images)
    all_images.update(_150ml_images)
    all_images.update(_200ml_images)
    all_images.update(_250ml_images)
    all_images.update(_300ml_images)
    all_images.update(_350ml_images)
    unknown_images.update(_A_images)
    unknown_images.update(_B_images)
    unknown_images.update(_C_images)
    
    column_names = ['Volume(in ml)', 'PixelCount']
    img_pixel_data = pd.DataFrame(columns=column_names)

    #   Compute pixel values of the fluid in each image via naive thresholding
    for directory, image_paths in all_images.items():
        for image_path in image_paths:
            img_white_pixel_count = img_pipeline(image_path)
            
            directory_int = int(directory[:-2])
            
            img_pixel_data.loc[len(img_pixel_data)] = [directory_int, img_white_pixel_count]
                      
            #print(f"{cropped_image.shape}: {img_white_pixel_count} units -> {directory_int}")
            #show_image('Image', cropped_image)

    model = LinearFunction()
    model.fit(img_pixel_data)
    print(f"Parameters:")
    model.show_parameters()
    print("\n")
    
    
    for directory, image_paths in unknown_images.items():
        print(f"Directory {directory}")
        for image_path in image_paths:
            img_white_pixel_count = img_pipeline(image_path)
            prediction = model.predict(img_white_pixel_count)
            print(f"{prediction}ml")
        print("\n")
                        
def img_pipeline(image_path):
>>>>>>> dec589ee5a3f1225552a3fe64072176dc8518b3a
    #   Pipeline: crop image as close to the bottle as possible -> threshold image to black out the fluid -> invert the threshold to have fluid as white regions
    cropped_image = cv2.imread(image_path)
    #   Preprocess image to crop out the bottle
    cropped_image = crop(cropped_image, x_lower=1500, x_upper=2000, y_lower=550, y_upper=1500)                    
    cropped_image = naive_threshold(cropped_image, 100, 225)
    cropped_image = extract_black_regions(cropped_image)
    return count_white_pixels(cropped_image)
        
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

#   Helper class for printing formatted text using ANSI color codes
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
if __name__ == "__main__":
    if "IS_VERBOSE" in [arg.upper() for arg in sys.argv]:
        IS_VERBOSE = True
    main()