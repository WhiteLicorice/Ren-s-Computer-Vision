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
    image1 = cv2.imread("IMG1.jpg")
    image2 = cv2.imread("IMG2.jpg")
    
    stitched_image = stitch_image(image1, image2)
    
    show_image("Stitched Image", stitched_image)
    
    cv2.imwrite("image_test.jpg", stitched_image)
    
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
    
if __name__ == "__main__":
    main()