import cv2
import numpy as np
import time

start = time.time()

# Load the images
image1 = cv2.imread('data/IMG1.jpg')
image2 = cv2.imread('data/IMG2.jpg')

height = int(image1.shape[0]) 
width = int(image1.shape[1])

# Pad an image with black pixels based on the shape of other image for simple cropping in final part
image2 = cv2.copyMakeBorder(image2, height, height, width, width, cv2.BORDER_CONSTANT, value=0)

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize the feature detector and extractor (e.g., SIFT)
descriptor = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = descriptor.detectAndCompute(gray1, None)
keypoints2, descriptors2 = descriptor.detectAndCompute(gray2, None)

# Initialize the feature matcher using FLANN matching
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# Match the descriptors using FLANN matching
matches = matcher.match(descriptors1, descriptors2)

# Select the top N matches
num_matches = 50
matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

# Extract matching keypoints
src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

# Estimate the homography matrix
homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10)

# Warp the first image using the homography
result = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

# Blending the warped image with the second image using alpha blending
alpha = 0.5 # blending factor
blended_image = cv2.addWeighted(result, alpha, image2, 1 - alpha, 0)

# Get end bounds of the blended image
lower = (1,1,1) # lower bound for each channel (nonblack pixels)
upper = (255,255,255) # upper bound for each channel

# Create the mask for white pixel finding
# Retrieves the pixels within the bounds as boolean, then .astype(np.uint8) * 255 makes it white for mask
mask = np.all((blended_image >= lower) & (blended_image <= upper), axis=2).astype(np.uint8) * 255

# Get white pixel bounds via the coordinates
white = np.where(mask==255)
xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])

# Crop image using the coordinates
result = blended_image[ymin:ymax, xmin:xmax]

end = time.time()
print("Time --->>>>>", end - start)

cv2.imwrite("blended_image.jpg", result)
# Display the blended image
cv2.namedWindow('Blended Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Blended Image', result.shape[1], result.shape[0])
cv2.imshow('Blended Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()