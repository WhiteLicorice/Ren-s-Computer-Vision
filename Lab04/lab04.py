"""
    Name: Computer Vision Laboratory 04
    Author(s): Rene Andre Bedonia Jocsing & Ron Gerlan Naragdao
    Date Modified: 03/08/2024 
    Usage: python lab04.py
    Description:
        The goal of this laboratory exercise is to estimate the amount of liquid contained in a bottle.
        The directory 'guess' contains images of the bottle with unknown amounts of liquid. You are to guess these amounts.
        OpenCV image filtering, thresholding, or morphology operations are allowed.
"""

import cv2
import numpy as np
import glob
import pandas as pd

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
    print(f"Parameters: {model.show_parameters()}\n")
    
    
    for directory, image_paths in unknown_images.items():
        print(f"Directory {directory}\n")
        for image_path in image_paths:
            img_white_pixel_count = img_pipeline(image_path)
            prediction = model.predict(img_white_pixel_count)
            print(f"{prediction}ml")
                        
def img_pipeline(image_path):
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
    
class LinearFunction:
    def __init__(self):
        self.slope = 0
        self.intercept = 0
    
    def fit(self, data):
        y = data["Volume(in ml)"].values  
        X = data["PixelCount"].values
        
        self.slope, self.intercept = np.polyfit(X, y, 1)

    def predict(self, input):
        return self.slope * input + self.intercept
        
    def show_parameters(self):
        print("Slope:", self.slope)
        print("Intercept:", self.intercept)
    

#   Helper method for collecting all the images in a directory  
def collect_images(input_directory):
    search_pattern = f"input/{input_directory}/*.jpg"
    image_files = glob.glob(search_pattern)
    return {input_directory: image_files}

if __name__ == "__main__":
    main()