import numpy as np
import cv2
from skimage.morphology import binary_closing, binary_opening

def cv2_stitch(img_list: list, method: str = 'perspective') -> np.ndarray:
    """
    Stitches an array of images together using the high level Stitcher class from cv2.
    Used as a reference to see the best output.

    Parameters:
            img_list: an array-like of images.
                
    Returns:
            stitched_img: the image stitched from the supplied images
    """
    
    stitcher = None
    
    if method == 'perspective':
        stitcher = cv2.Stitcher_create()
    elif method == 'affine':
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    else:
        raise ValueError("Invalid stitching method. Supported methods are 'perspective' and 'affine'.")

    status, stitched_img = stitcher.stitch(img_list)

    if status == cv2.Stitcher_OK:
        return stitched_img
    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        raise Exception("Insufficient images provided for stitching.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        raise Exception("Homography estimation failed during stitching.")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        raise Exception("Camera parameter adjustment failed during stitching.")
    else:
        raise Exception(f"Error in stitching image via cv2 stitcher has occurred: {status}")
    
def convert_to_gray(image: np.ndarray) -> tuple:
    """
    Converts the input images to grayscale.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        gray_image: The input image in greyscale.
    """
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image
def extract_sift(image: np.ndarray) -> tuple:
    """
    Extracts keypoints and descriptors from the input image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        keypoints, descriptors: the keypoints and descriptors of the image.
    """
    
    descriptor = cv2.SIFT_create()
    keypoints, descriptors = descriptor.detectAndCompute(image, None)
    
    return keypoints, descriptors

def match_descriptors(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    matcher_type : str = 'bruteforce') -> list:
    """
    Matches descriptors between two sets using a specified matcher.

    Parameters:
        descriptors1 (numpy.ndarray): Descriptors from the first set of keypoints.
        descriptors2 (numpy.ndarray): Descriptors from the second set of keypoints.
        matcher_type (string): The type of matcher to use.
        
    Returns:
        matches: List of matches.
    """
    
    matcher = None
    
    match matcher_type:
        case 'bruteforce': 
            #   NORM_L2 should be used with BFMatcher according to cv2 documenations
            #   crossCheck returns only consistent pair of matches, serving as an alternative to Lowe's ratio test
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        case 'flann':
            #   Use autotuned to let cv2 decide the best algorithm based on the dataset
            index_params = dict(algorithm="autotuned", trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
    matches = matcher.match(descriptors1, descriptors2)
    return matches

def select_top_matches(matches: list, num_matches: int = 50) -> list:
    """
    Selects the top N matches based on distance.

    Parameters:
        matches (list): List of matches.
        num_matches (int): Number of top matches to select.

    Returns:
        list: List of selected matches.
    """
    good_matches = sorted(matches, key=lambda match: match.distance)[:num_matches]
    return good_matches

def estimate_homography(keypoints1: list, keypoints2: list, selected_matches: list, ransac_threshold: int = 10) -> np.ndarray:
    """
    Estimates the homography matrix based on selected matches.

    Parameters:
        keypoints1 (list): Keypoints from the first image.
        keypoints2 (list): Keypoints from the second image.
        selected_matches (list): List of selected matches.
        RANSAC_threshold (int): The threshold of the RANSAC algorithm.
    Returns:
        numpy.ndarray: Homography matrix.
    """
    src_points = np.float32([keypoints1[match.queryIdx].pt for match in selected_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[match.trainIdx].pt for match in selected_matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransacReprojThreshold=ransac_threshold)
    #print(homography)
    
    return homography

def blend_images(image1: np.ndarray, image2: np.ndarray, homography: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blends two images using alpha blending and homography transformation.

    Parameters:
        image1 (numpy.ndarray): First input image.
        image2 (numpy.ndarray): Second input image.
        homography (numpy.ndarray): Homography matrix.
        alpha (float): Blending factor for alpha blending (default is 0.5).

    Returns:
        numpy.ndarray: Blended image.
    """
 
    result = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))
    
    #   Get the intersection
    mask = (result != 0) & (image2 != 0)
    #   Refine the mask and remove any discrepancies
    mask = binary_closing(mask)
    mask = binary_opening(mask)
    
    #   Make a grey mask
    mask = mask.astype(np.uint8) * 255
    
    #   Normalize the images
    mask = mask / 255.0
    result = result / 255.0
    image2 = image2 / 255.0
    
    #   Remove the part of image where mask is
    for i in range(3):
        image2[:,:,i] *= (1 - mask[:,:,0])

    blended_image = cv2.addWeighted(result, 1, image2, 1, 0)
    
    #   Denormalize image
    return (blended_image * 255).clip(0, 255).astype(np.uint8)

def bound_image(blended_image: np.ndarray) -> np.ndarray:
    """
    Bounds the blended image based on the white pixels in the mask.

    Parameters:
        blended_image (numpy.ndarray): The unbounded blended image.

    Returns:
        numpy.ndarray: The bounded image.
    """

    #   Create mask for white pixel finding
    #   Define upper and lower bounds for each channel
    lower = (1, 1, 1)
    upper = (255, 255, 255)
    
    #   Create the mask for white pixel finding
    #   Retrieve the pixels within the bounds as boolean, then .astype(np.uint8) * 255 makes it white for mask
    mask = cv2.inRange(blended_image, lower, upper)

    #   Get white pixel bounds via the coordinates
    white = np.where(mask == 255)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])

    #   Bound image using the coordinates
    bound_image = blended_image[ymin:ymax, xmin:xmax]
    return bound_image

def pad_image(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Pads the second image with black pixels based on the shape of the first image.

    Parameters:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image.

    Returns:
        numpy.ndarray: Padded image.
    """
    height = int(image1.shape[0]) 
    width = int(image1.shape[1])

    #   Create a black padded image based on the first image, with the second image at the center
    padded_image = cv2.copyMakeBorder(image2, 
                                       height, height, 
                                       width, width, 
                                       cv2.BORDER_CONSTANT, value=0)
    return padded_image

def stitch_image(image1: np.ndarray, image2: np.ndarray, num_matches: int = 50, alpha: float = 0.9) -> None:
    """
    Blends two images using keypoint matching and homography estimation.

    Parameters:
        image1 (image): The first input image.
        image2 (image): The second input image.
        num_matches (int): Number of matches to consider for homography estimation (default is 50).
        alpha (float): Blending factor for alpha blending (default is 0.5).
    Returns:
        stitched_image: the two images stitched along common features.
    """
    
    #   Pad an image with black pixels based on the shape of other image for simple cropping in final part
    image2 = pad_image(image1, image2)

    #   Convert to grayscale
    gray1 = convert_to_gray(image1)
    gray2 = convert_to_gray(image2)

    #   Extract keypoints and descriptors
    keypoints1, descriptors1 = extract_sift(gray1) 
    keypoints2, descriptors2 = extract_sift(gray2)

    #   Match descriptors
    matches = match_descriptors(descriptors1, descriptors2)

    #   Select top matches
    selected_matches = select_top_matches(matches, num_matches)

    #   Estimate homography
    homography = estimate_homography(keypoints1, keypoints2, selected_matches)
    
    ##  TODO: Diagnose this line
    # #   Skip space themed warp
    # if homography[1][1] < 0 or homography[1][1] > 2:
    #     return None

    #   Blend images
    blended_image = blend_images(image1, image2, homography, alpha)
    
    #   Fetch final stitched image
    stitched_image = bound_image(blended_image)

    assert stitched_image is not None, "Failed to stitch image. Perhaps matches are insufficient?"
    
    return stitched_image

#   Imported as is from Lab04/thresholding.py
def crop(image: np.ndarray, 
         x_lower: int = None, 
         x_upper: int = None, 
         y_lower: int = None, 
         y_upper: int = None) -> np.ndarray:
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

#   Generic cv2 function to extract frames from video
def extract_frames(video_path, frame_interval):
    """
    Extracts frames from a video at a specified interval.

    Parameters:
        video_path (str): Path to the input video file.
        frame_interval (int): Interval at which frames are extracted.

    Returns:
        List[np.ndarray]: List of extracted frames as NumPy arrays.
    """
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = [ ]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)
        
        frame_count += 1

    cap.release()
    
    return frames

def generate_action_masks(frames: list, background_subtractor: str = 'mog') -> np.ndarray:
    """
    Generates action masks using background subtraction.

    Parameters:
        frames (List[np.ndarray]): List of input frames.
        background_subtractor (str): Background subtractor algorithm ('knn' or 'mog').

    Returns:
        List[np.ndarray]: List of action masks.
    """
    
    processed_frames = [ ]
    bg_sub = None
    match background_subtractor:
        case 'knn': bg_sub = cv2.createBackgroundSubtractorKNN()
        case 'mog': bg_sub = cv2.createBackgroundSubtractorMOG2()
        case _: raise Exception ("Invalid background subtractor object. Valid arguments are 'knn' and 'mog'." )
    
    for frame in frames:
        processed_frame = bg_sub.apply(frame)
        processed_frames.append(processed_frame)
        
    return processed_frames

def generate_action_shot(frames: list):
    """
    Generates an action shot by overlaying moving objects on the background.

    Parameters:
        frames (List[np.ndarray]): List of input frames.

    Returns:
        np.ndarray: Action shot with overlaid moving objects.
    """
    #   Read and process the reference frame, assuming that the reference frame is at index 0
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

    ##  Implement modified version of generate_action_masks() here
    
    #   Initialize KNN background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    # Compute the median of all frames to estimate the background
    background = np.median(frames, axis=0).astype(np.uint8)

    #   Apply background subtraction and overlay the moving objects on the background using multi-layer blending
    aligned_frames = [reference_frame] + aligned_frames  # Include reference frame
    blended_result = background.copy()
    #i = 0
    
    #   Apply background subtraction to aligned frames
    for frame in aligned_frames:
        fgmask = fgbg.apply(frame)

        #   Create a mask for the current action frame
        action_mask = np.zeros_like(background)
        action_mask[fgmask > 0] = 255
        
        #   Kernel for eroding and dilating        
        kernelSmall = np.ones((3, 3), np.uint8)
        #   Kernel for filling up holes
        kernelLarge = np.ones((30, 30), np.uint8)
        
        #   Remove any thin lines (Not always guaranteed)
        action_mask = cv2.erode(action_mask, kernelSmall, iterations=2)
        action_mask = cv2.dilate(action_mask, kernelSmall, iterations=2)
        
        #   Fill up any space in between (Not always guaranteed to be correct with original)
        action_mask = cv2.morphologyEx(action_mask, cv2.MORPH_CLOSE, kernelLarge)
        
        #   Normalize the images
        action_mask = action_mask / 255.0
        blended_result = blended_result / 255.0
        frame = frame / 255.0
        
        #   Remove the part of image where mask is
        for j in range(3):
            blended_result[:,:,j] *= (1 - action_mask[:,:,j])
            frame[:,:,j] *= (action_mask[:,:,j])

        #   Blend the result and the frame with denormalizing
        blended_result = cv2.addWeighted(blended_result, 1, frame, 1, 0)
        blended_result = (blended_result * 255).clip(0, 255).astype(np.uint8)

        #show_image('Action Shot', blended_result)
        #save_image(blended_result, f"action_{i}")
        #i += 1
        
    return blended_result