import numpy as np
import cv2

def cv2_stitch(img_list: list) -> np.ndarray:
    imgStitcher = cv2.Stitcher_create()
    error, stitched_img = imgStitcher.stitch(img_list)
    
    if not error:
        return stitched_img
    else: 
        raise Exception(f"Error in stitching image via cv2 stitcher has occured: {error}")
    
#   TODO: Perhaps use homography matrices and SURF?