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
    
    The testing suite can be accessed in lab05.py and the relevant functions should be accessed via stitching.py.
"""

import cv2
import glob
import matplotlib.pyplot as plt
import time
import os

from stitching import *

def main():
    """IMAGE STITCHING"""
    #test_cv2_stitcher()        #   Passing
    #test_pad_image()           #   Passing
    #test_extract_features()    #   Passing
    #test_find_matches()        #   Passing
    #test_find_best_matches()   #   Passing
    test_image_stitching()    #   Passing
    
    """ACTION SHOT"""
    #test_extract_frames()      #   Passing
    #test_action_mask()         #   Passing
    #test_action_shot()         #   Passing

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

@DeprecationWarning
def bulk_image_stitching():
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

def test_image_stitching():
    start_time = time.time()
    
    img1 = cv2.imread("data/IMG1.jpg")
    img2 = cv2.imread("data/IMG2.jpg")
    img3 = cv2.imread("data/IMG3.jpg")
    
    img4 = cv2.imread("data/IMG4.jpg")
    img5 = cv2.imread("data/IMG5.jpg")
    img6 = cv2.imread("data/IMG6.jpg")
    
    img7 = cv2.imread("data/IMG7.jpg")
    img8 = cv2.imread("data/IMG8.jpg")
    img9 = cv2.imread("data/IMG9.jpg")
    
    # show_image("Image 1", img1)
    # show_image("Image 2", img2)
    # show_image("Image 3", img3)
    # show_image("Image 4", img4)
    # show_image("Image 5", img5)
    # show_image("Image 6", img6)
    # show_image("Image 7", img7)
    # show_image("Image 8", img8)
    # show_image("Image 9", img9)

    img1_2_3 = cv2.imread("output/img123.jpg")
    if img1_2_3 is None:
        img1_2_3 = stitch_image(stitch_image(img2, img3, num_matches=1000), img1, num_matches=1000)
        save_image(img1_2_3, "img123")
    show_image("Image 1, 2, 3", img1_2_3) 
    
    img4_5_6 = cv2.imread("output/img456.jpg")
    if img4_5_6 is None:
        img4_5_6 = stitch_image(stitch_image(img4, img5), img6)
        save_image(img4_5_6, "img456")
    show_image("Image 4, 5, 6", img4_5_6)
    
    img7_8_9 = cv2.imread("output/img789.jpg")
    if img7_8_9 is None:
        img7_8_9 = stitch_image(stitch_image(img7, img8), img9)
        save_image(img7_8_9, "img789")
    show_image("Image 7, 8, 9", img7_8_9)
    
    #   Band-aid solution to out of memory errors by Ren TM
    img1_2_3 = resize_image(img1_2_3, 50)
    img4_5_6 = resize_image(img4_5_6, 50)
    img7_8_9 = resize_image(img7_8_9, 50)
    
    final_image = stitch_image(stitch_image(img1_2_3, img4_5_6, num_matches=5000), img7_8_9, num_matches=5000)
    show_image("Final Image", final_image)
    save_image(final_image, "stitched_image")
    
    total_time = time.time() - start_time
    
    print(f"Stitching Duration: {total_time}")
    
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

def test_extract_frames():
    video = "data/spike.mp4"
    interval = 15
    frames = extract_frames(video, interval)
    i = 0
    for frame in frames:
        show_image(f'action_{i}', frame)
        i += interval

def test_action_mask():
    mask_frames = generate_action_masks(extract_frames("data/spike.mp4", 15), 'mog')
    for frame in mask_frames:
        show_image('Mask Frame', frame)

def test_action_shot():
    frames = extract_frames("data/spike.mp4", 15)
    action_shot = generate_action_shot(extract_frames("data/spike.mp4", 15))
    final_shot = crop(action_shot, x_upper=1100)
    
    save_image(final_shot, 'action_shot')
    show_image("Action Shot", final_shot)
    
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

def save_image(image, file_name, target_path='output', extension='jpg'):
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

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

if __name__ == "__main__":
    main()