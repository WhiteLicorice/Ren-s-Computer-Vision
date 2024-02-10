"""
    Name: Computer Vision Laboratory 02
    Author: Rene Andre Bedonia Jocsing
    Date Modified: 02/11/2024 
    Usage: python lab02.py
    Description:
        This is a Python script that utilizes the cv2 package to implement an image filtering function
        and use it to create hybrid images using a simplified version of the SIGGRAPH 2006
        paper by Oliva, Torralba, and Schyns. Hybrid images are static images that change
        in interpretation as a function of the viewing distance. The basic idea is that high
        frequency tends to dominate perception when it is available, but, at a distance, only
        the low frequency (smooth) part of the signal can be seen. By blending the high frequency
        portion of one image with the low-frequency portion of another, you get a hybrid image 
        that leads to different interpretations at different distances.
        All the necessary image filtering functions can be found in hybrid.py.
        The testing suite can be found in lab02.py and should be used to access hybrid.py.
"""

from hybrid import cross_correlation, convolution, gaussian_blur, low_pass, high_pass, create_hybrid_image

import cv2
import numpy as np

def main():
    test_create_hybrid_image()
    #   TODO: Build a flet app to accept images and control filters?

'''UTILITIES'''
#   Helper method to show an image
def show_image(name, image):
    #   Show image in a normal window
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''TESTING'''

def test_correlation():
    #   Assume that image.png already exists
    img = cv2.imread('input/image.png')
    
    show_image('Samurai Doge', img)
    
    #   Mean Filter Kernel
    #kernel = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)

    #   Horizontal Sobel Kernel for Edge Detection
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    #   Vertical Sobel Kernel for Edge Detection
    #kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
 
    # Apply cross-correlation
    result = cross_correlation(img, kernel)

    show_image('Samurai Doge', result)

def test_convolution():
    #   Assume that image.png already exists
    img = cv2.imread('input/image.png')  # Assuming grayscale image
    
    show_image('Samurai Doge', img)
    
    #   Mean Filter Kernel
    #kernel = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9)

    #   Horizontal Sobel Kernel for Edge Detection
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    #   Vertical Sobel Kernel for Edge Detection
    #kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    # Apply convolution with random Gaussian blur kernel
    result = convolution(img, gaussian_blur(5, 3, 3))

    show_image('Samurai Doge', result)

def test_low_pass():
    img = cv2.imread('input/image.png')
    show_image('Samurai Doge', img)
    
    # Apply convolution with random Gaussian blur kernel
    result = low_pass(img, 5, 5)
    
    show_image('Samurai Low Pass', result)

def test_high_pass():
    img = cv2.imread('input/image.png')
    show_image('Samurai Doge', img)
    
    result = high_pass(img, 5, 5)
    
    show_image('Samurai High Pass', result)

def test_create_hybrid_image():
    #   Assuming we have two images: img1 and img2
    #       img1: The image for the low-frequency component
    #       img2: The image for the high-frequency component

    img1 = cv2.imread('input/image1.png')
    img2 = cv2.imread('input/image2.png')
    
    show_image('Honda', img1)
    show_image('Ferrari', img2)
    
    #   Define parameters for the low-frequency component
    sigma1 = 5  #   Sigma for Gaussian blur
    size1 = 5   #   Size of the kernel
    high_low1 = 'low'  #    Type of filtering ('low' for low-pass, 'high' for high-pass)

    #   Define parameters for the high-frequency component
    sigma2 = 1  #   Sigma for Gaussian blur
    size2 = 5   #   Size of the kernel
    high_low2 = 'high'  #   Type of filtering ('low' for low-pass, 'high' for high-pass)

    #   Mixing ratio
    mixin_ratio = 0.5

    #   Scale factor
    scale_factor = 1.0

    #   Create the hybrid image
    hybrid_image = create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio, scale_factor)
    
    show_image('Hondararri', hybrid_image)

if __name__ == "__main__":
    main()

"""
=HINTS====================================
You may find the following code snippets useful

"""

''' 
# mean filter kernel
kernel = np.array([(1,1,1),(1,1,1),(1,1,1)])*(1/9) # 3x3

#Gaussian kernel
size=5
sigma=3
center=int(size/2)
kernel=np.zeros((size,size))
for i in range(size):
	for j in range(size):
          kernel[i,j] = (1/(2*np.pi*sigma**2))*np.exp(-((i-center)**2+(j-center)**2)/(2*sigma**2))
kernel=kernel/np.sum(kernel)	#Normalize values so that sum is 1.0

#dimensions of the image and the kernel
image_height, image_width = imageGray.shape
kernel_height, kernel_width = ______________________

#Padding
imagePadded = np.zeros((image_height+kernel_height-1,________________________)) # zero-padding scheme, you may opt for other schemes
for i in range(image_height):
	for j in range(image_width):
		imagePadded[i+int((kernel_height-1)/2), j+_____________________] = imageGray[i,j]  #copy Image to padded array

 
#correlation
for i in range(________):
	for j in range(image_width):
		window = imagePadded[____________________, j:j+kernel_width]
		imageGray[i,j] = np.sum(window*kernel)  #numpy does element-wise multiplication on arrays

#convolution
		#np.flip(kernel)  # flips horizontally and vertically
		#correlation

#low pass filter
	#Either convolution or correlation using Gaussian kernel will do 
	#since Gaussian kernel is all-axis symmetric, either correlation or convolution can be used
	
#high pass filter
	# original image - low pass image

'''

"""
#merge two images (low pass image + high pass image)
	alpha*Image1 + (1-alpha)Image2   # alpha is the amount of blending between the two images
"""

