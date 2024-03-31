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

from stitching import cv2_stitch


def main():
    test_cv2_stitcher()     #   Passing
    #   TODO: Devise a strategy that doesn't use cv2.stitcher(), using the results of test_cv2_stitcher as reference
    
"""UTILITIES"""
def test_cv2_stitcher():
    img_list = fetch_images()
    stitched_image = cv2_stitch(img_list)
    show_image("stitched_image_cv2_stitcher", stitched_image)
    
def show_image(image_label, image):
    #   Show image in a normal window
    cv2.namedWindow(image_label, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_label, image.shape[1], image.shape[0])
    cv2.imshow(image_label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fetch_images():
    img_paths = glob.glob("data/*.jpg")
    imgs = []
    
    for img in img_paths:
        img_read = cv2.imread(img)
        imgs.append(img_read)
        
    return imgs
    
if __name__ == "__main__":
    main()