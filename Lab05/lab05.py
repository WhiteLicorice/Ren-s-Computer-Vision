"""
    Name: Computer Vision Laboratory 05
    Author(s): Rene Andre Bedonia Jocsing & Ron Gerlan Naragdao
    Date Modified: 03/31/2024 
    Usage: python lab05.py
    Description:
        1. Read https://medium.com/@paulsonpremsingh7/image-stitching-using-opencv-a-step-by-step-tutorial-9214aa4255ec

    2. Using the code in 1) as basis, stitch the images from a directory named 'data'.

    3. Using the video named 'spike.mp4' in directory 'data', generate an actionshot image. 
    Actionshot is a method of capturing an object in action and displaying it in a single image with multiple sequential appearances of the object.
    Extra credits for using your own video for the actionshot image.
    (doing the cartwheel, running forehand in tennis, grand jeté in ballet, somersault in acrobatics or diving, forward flip in skateboarding)
    SAFETY FIRST. Be sure you know what you are doing should you embark in this adventure.
    
    The testing suite can be accessed in lab05.py and the stitching functions should be accessed via stitching.py.
"""

import cv2
import glob
import matplotlib.pyplot as plt
import time
import os

from stitching import *


def main():
    #test_cv2_stitcher()        #   Passing
    #test_pad_image()           #   Passing
    #test_extract_features()    #   Passing
    #test_find_matches()        #   Passing
    #test_find_best_matches()   #   Passing
    test_image_stitching()     #   Passing
    pass

"""TESTING SUITE"""
def test_cv2_stitcher():
    img_list = fetch_images(directory="data", file_type="jpg")
    stitched_image = cv2_stitch(img_list)
    show_image("stitched_image_cv2_stitcher", stitched_image)
    cv2.imwrite("stitched_image.jpg", stitched_image)

def test_find_matches():
    image1 = cv2.imread("IMG1.jpg")
    image2 = cv2.imread("IMG2.jpg")

    #   Extract keypoints and descriptors from the images
    keypoints1, descriptors1 = extract_sift(image1)
    keypoints2, descriptors2 = extract_sift(image2)

    #   Match the descriptors
    matches = match_descriptors(descriptors1, descriptors2, index_algorithm="autotuned", num_trees=5, num_checks=50)

    #   Draw matches on a new image
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plot_image("Matches", matched_image)
    
def test_find_best_matches():
    image1 = cv2.imread("IMG1.jpg")
    image2 = cv2.imread("IMG2.jpg")

    #   Extract keypoints and descriptors from the images
    keypoints1, descriptors1 = extract_sift(image1)
    keypoints2, descriptors2 = extract_sift(image2)

    #   Match the descriptors
    matches = match_descriptors(descriptors1, descriptors2, index_algorithm="autotuned", num_trees=5, num_checks=50)

    top_matches = select_top_matches(matches, 50)
    
    #   Draw matches on a new image
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, top_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plot_image("Matches", matched_image)
    
def test_image_stitching():
    img_ratio = 1
    start = time.time()
    
    #   Fetch images
    img_list = fetch_images(directory="data", file_type="jpg")

    #   New processed image (if img_ratio is not set to 1)
    new_img_list = []
    for i in range(len(img_list)):
        height, width = img_list[i].shape[:2]
        width = int(width/img_ratio)
        height = int(height/img_ratio)
        img = cv2.resize(img_list[i], (width,height))
        new_img_list.append(img)
    
    #   Initial stitched image
    stitched_image = stitch_image(new_img_list[0], new_img_list[1])
    #   Index for while loop    
    i = 2
    
    while i != len(new_img_list):
        stitched_image = stitch_image(new_img_list[i], stitched_image)
        #   Place the unmatched image at the end of list
        if stitched_image is None:
            skip = new_img_list.pop(i)
            new_img_list.append(skip)
            #   Load saved point
            stitched_image = cv2.imread(f"results/image_test{i-1}.jpg")
            #print("Skip")
        #   Proceed with next image
        else:
            #   Save point
            cv2.imwrite(f"results/image_test{i}.jpg", stitched_image)
            i+=1
           
    end = time.time()
    print("Time elapsed: ", end - start)

    show_image("Stitched Image", stitched_image)
    save_image(stitched_image, "image_test")
    
def test_pad_image():
    image1 = cv2.imread("IMG1.jpg")
    image2 = cv2.imread("IMG2.jpg")
    
    padded_image = pad_image(image1, image2)
    
    show_image("Padded Image", padded_image)

def test_extract_features():
    image = cv2.imread("IMG1.jpg")

    #   Extract keypoints and descriptors
    keypoints, _ = extract_sift(image)

    #   Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

    plot_image("Image Features", image_with_keypoints)

"""UTILITIES"""
def show_image(image_label, image):
    #   Show image in a normal window
    cv2.namedWindow(image_label, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_label, image.shape[1], image.shape[0])
    cv2.imshow(image_label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def plot_image(plot_label, image):
    # Display the image with keypoints
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(plot_label)
    plt.axis('off')
    plt.show()

def fetch_images(directory, file_type):
    img_paths = glob.glob(f"{directory}/*.{file_type}")
    imgs = []
    
    for img in img_paths:
        img_read = cv2.imread(img)
        imgs.append(img_read)
        
    return imgs

def delete_images(directory, file_type):
    files = glob.glob(f'{directory}/*.{file_type}')
    for f in files:
        os.remove(f)

def save_image(image, file_name, target_path="output", extension='jpg'):
    # Ensure target directory exists
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"Directory '{target_path}' didn't exist, created it.")

    # Write the image
    try:
        cv2.imwrite(f"{target_path}/{file_name}.{extension}", image)
        print(f"Image '{file_name}.{extension}' saved successfully to '{target_path}' directory.")
    except Exception as e:
        print(f"Error occurred while saving the image: {e}")

#   Imported as is from Lab04/thresholding.py
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
    
if __name__ == "__main__":
    main()