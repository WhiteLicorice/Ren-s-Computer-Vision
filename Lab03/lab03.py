"""
    Name: Computer Vision Laboratory 03
    Author: Rene Andre Bedonia Jocsing
    Date Modified: 02/26/2024 
    Usage: python lab03.py
    Description:
        This is a Python script that utilizes the cv2 package to implement an image blending
        algorithm using Laplacian and Gaussian pyramids.
        All the necessary functions can be found in blend.py.
        The testing suite can be found in lab03.py and should be used to access blend.py.
"""

import cv2
import numpy as np

from blending import interpolate, decimate, construct_gaussian_pyramid,construct_pyramids, blend_pyramids, blend_image, reconstruct_image, bound

def main():
    #test_interpolate()                     #   Passing        
    #test_decimate()                        #   Passing
    #test_construct_gaussian_pyramids()     #   Passing
    #test_construct_pyramids()              #   Passing
    #test_blend_pyramids()                  #   Passing
    test_reconstruct_image()               #   Passing
    #test_blend_image()                     #   Passing
    #test_blend_image_creative()            #   Passing
    return
    
#   Helper method to show an image
def show_image(name, image):
    #   Show image in a normal window
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#   Helper method for saving strips (key is used as a filename identifier)
def save_images(images, key):
    #   Save and/or display each strip
    for i, strip in enumerate(images):
        cv2.imwrite(f'output/{key}_{i + 1}.jpg', strip)  #    Save each strip to a uniquely named file

#   Helper method for saving a single image  
def save_image(image, key):
    cv2.imwrite(f'output/{key}.jpg', image)

#   Helper method for naive brightening/dimming of image via multiplying each pixel by a brightness factor
def brighten_image(image, brightness_factor):
    return np.clip(image.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

"""TESTING MODULES"""  
def test_interpolate():
    image = cv2.imread("input/image.png")
    
    print(image.shape)
    show_image("Original", image)
    
    interpolated_image = bound(interpolate(image))
    
    print(interpolated_image.shape)
    show_image("Interpolated", interpolated_image)
    
def test_decimate():
    image = cv2.imread("input/image.png")
    
    print(image.shape)
    show_image("Original", image)
    
    decimated_image = bound(decimate(image))
    
    print(decimated_image.shape)
    show_image("Decimated", decimated_image)
    
def test_construct_gaussian_pyramids():
    image = cv2.imread("input/apple.jpg")
    
    #print(f"Gaussian 0: {image.shape}")
    #show_image("Gaussian 0", image)
    
    gaussian_pyramid = construct_gaussian_pyramid(image)
    
    #save_images(gaussian_pyramid, "gaussian")
    
    for i, level in enumerate(gaussian_pyramid):
        print(f"Gaussian {i + 1}: {level.shape}")
        show_image(f"Gaussian {i + 1}", bound(level))
        
def test_construct_pyramids():
    # Read the input image
    image = cv2.imread("input/apple.jpg")

    #print(f"Laplacian 0: {image.shape}")
    show_image("Original", image)
    
    # Construct the Laplacian pyramid
    laplacian_pyramid, gaussian_pyramid = construct_pyramids(image)
    
    #save_images(laplacian_pyramid, "laplacian")
    #save_images(gaussian_pyramid, "gaussian")

    # Display the Gaussian pyramid
    for i, level in enumerate(gaussian_pyramid):
        print(f"Gaussian {i + 1}: {level.shape}")
        show_image(f"Gaussian {i + 1}", bound(level))
    
    # Display the Laplacian pyramid
    for i, level in enumerate(laplacian_pyramid):
        print(f"Laplacian {i + 1}: {level.shape}")
        show_image(f"Laplacian {i + 1}", bound(level))
        
def test_blend_pyramids():
    # Read the input image
    A = cv2.imread("input/orange.jpg")
    B = cv2.imread("input/apple.jpg")
    M = cv2.imread("input/mask.jpg")

    show_image("Orange", A)
    show_image("Apple", B)
    
    blended_pyramid = blend_pyramids(A, B, M)
    
    #save_images(blended_pyramid, "blended")

    # Display the Gaussian pyramid
    for i, level in enumerate(blended_pyramid):
        print(f"Blended {i + 1}: {level.shape}")
        show_image(f"Blended {i + 1}", level)
        
def test_reconstruct_image():
    # Read the input image
    image = cv2.imread("input/orange.jpg")
    
    show_image("Original Image", image)
    
    laplacian_pyramids, _ = construct_pyramids(image)
    
    # for i, level in enumerate(laplacian_pyramids):
    #     print(f"Laplacian {i + 1}: {level.shape}")
    #     show_image(f"Laplacian {i + 1}", level)
    
    reconstructed_image = reconstruct_image(laplacian_pyramids)
    
    show_image("Reconstructed Image", reconstructed_image)
    
def test_blend_image():
    # Read the images, A and B, as well as the mask M (we assume they are already of the same dimensions)
    A = cv2.imread("input/orange.jpg", cv2.IMREAD_COLOR)
    B = cv2.imread("input/apple.jpg", cv2.IMREAD_COLOR)
    M = cv2.imread("input/mask.jpg", cv2.IMREAD_COLOR)

    show_image("Orange", A)
    show_image("Apple", B)

    blended_image = blend_image(A, B, M)
    
    #save_image(blended_image, "blended")

    #   Brighten image for viewing purposes
    #blended_image = brighten_image(blended_image, 1.5)
    
    #save_image(blended_image, "blended")
    
    show_image("Blended Image", blended_image)
    
def test_blend_image_creative():
    # Read the images, A and B, as well as the mask M (we assume they are already of the same dimensions)
    A = cv2.imread("input/pawn.png", cv2.IMREAD_COLOR)
    B = cv2.imread("input/bishop.png", cv2.IMREAD_COLOR)
    M = cv2.imread("input/crazymask.png", cv2.IMREAD_COLOR)

    show_image("Pawn", A)
    show_image("Bishop", B)

    blended_image = blend_image(A, B, M)
    
    #save_image(blended_image, "crazy")

    #   Brighten image for viewing purposes
    #blended_image = brighten_image(blended_image, 1.5)
    
    #save_image(blended_image, "crazy")
    
    show_image("Crazy Image", blended_image)
 
if __name__ == "__main__":
    main()