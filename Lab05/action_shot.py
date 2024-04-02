import cv2
import glob
import os
import numpy as np

from stitching import *

from lab05 import fetch_images, save_image, delete_images, show_image  ##  TODO: Remove these imports upon creation of final API

def main():
    delete_images('output', 'jpg')
    #test_action_frames()    #   Passing
    #test_action_mask()      #   Passing
    #test_action_shot()     #    Failing
    test_complete_action_shot()
    pass

#   Generic cv2 function to extract frames from video
def extract_frames(video_path, frame_interval):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = [ ]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            #save_image(frame, f'frame_{frame_count}')
            frames.append(frame)
        
        frame_count += 1

    cap.release()
    
    return frames
    
def generate_action_frames(frames: list, background_subtractor: str = 'knn') -> np.ndarray:
    processed_frames = [ ]
    bg_sub = None
    match background_subtractor:
        case 'knn': bg_sub = cv2.createBackgroundSubtractorKNN()
        case 'mog': bg_sub = cv2.createBackgroundSubtractorMOG2()
        case _: raise Exception ("Invalid background subtractor object. Valid arguments are 'knn' and 'mog'." )
    
    for frame in frames:
        processed_frame = bg_sub.apply(frame)
        #show_image("An Action Frame", processed_frame)
        processed_frames.append(processed_frame)
        
    return processed_frames

@DeprecationWarning
def create_action_shot(frames):
    #   Load frames
    #action_frames = generate_action_frames(frames, background_subtractor='knn')
    action_frames = [frames[0], ]
    # #   Create background subtractor
    background_subtractor = cv2.createBackgroundSubtractorKNN()

    #Apply background subtraction to each frame
    for frame in frames[1:]:
        fg_mask = background_subtractor.apply(frame)
        action_frames.append(cv2.bitwise_and(frame, frame, mask=fg_mask))

    #   Overlay action frames onto the first frame
    #   Use the first frame as the background
    action_shot = frames[0].copy()  
    
    i = 1
    for action_frame in action_frames[1:]:
        #save_image(action_frame, f'action_{i}')
        action_shot = cv2.add(action_shot, action_frame)  #     Overlay action frame onto the background
        i += 1

    return action_shot

def test_action_shot():
    video_path = "data/spike.mp4"
    frame_interval = 1

    delete_images('output', 'jpg')

    frames = extract_frames(video_path, frame_interval)

    action_shot = create_action_shot(frames)

    save_image(action_shot, 'action_shot')
    
def test_action_frames():
    video = "data/spike.mp4"
    interval = 1
    frames = extract_frames(video, interval)
    i = 0
    for frame in frames:
        save_image(frame, f'action_{i}')
        i += interval

def test_action_mask():
    video = "data/spike.mp4"
    interval = 1
    frames = extract_frames(video, interval)
    action_frames = generate_action_frames(frames, 'knn')
    
    i = 0
    for frame in action_frames:
        save_image(frame, f'mask_{i}')
        i += interval

def test_complete_action_shot():
    interval = 15
    frames = extract_frames("data/spike.mp4", interval)

    #   Read and process the reference frame
    reference_frame = frames[0]
    keypoints_ref, descriptors_ref = extract_sift(reference_frame)

    #   List to store aligned frames
    aligned_frames = [ ]

    #   Read and process subsequent frames
    for frame in frames[1:]:
        #   Detect keypoints and compute descriptors for the current frame
        keypoints_curr, descriptors_curr = extract_sift(frame)

        #   Match descriptors between reference frame and current frame
        matches = match_descriptors(descriptors_ref, descriptors_curr)
                
        good_matches = select_top_matches(matches, 50)

        #   Estimate homography using RANSAC
        if len(good_matches) > 10:
            homography = estimate_homography(keypoints_ref, keypoints_curr, good_matches)
            #   Warp current frame to align with reference frame
            aligned_frame = cv2.warpPerspective(frame, homography, (reference_frame.shape[1], reference_frame.shape[0]))
            aligned_frames.append(aligned_frame)

    ##  Begin overlaying frames
    
    #   Initialize KNN background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    # Compute the median of aligned frames to estimate the background
    aligned_frames = [reference_frame] + aligned_frames  # Include reference frame
    background = np.median(aligned_frames, axis=0).astype(np.uint8)

    i = 0
    # Apply background subtraction to aligned frames
    for frame in aligned_frames:
        fgmask = fgbg.apply(frame)

        #   Create resulting shot by overlaying the moving object on the background TODO: Use this line for MOG2
        result = cv2.copyTo(frame, fgmask, background)

        #   Apply the mask to the frame to achieve background subtraction # TODO: Use this line for KNN, quite buggy
        #result = cv2.bitwise_and(frame, frame, mask=fgmask)
        
        show_image('Action Shot', result)
        save_image(result, f"action_{i}")
        i+=1
        

if __name__ == '__main__':
    main()