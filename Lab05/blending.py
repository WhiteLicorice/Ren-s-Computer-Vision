import numpy as np
import cv2

def gaussian_stack(img, sigma = 5):
    
    g_copy = img.copy()
    g_array = [g_copy]
    for i in range(0, 6):
        # Sigma=(x,y,z) dimensions to preserve color
        G = cv2.pyrDown(g_array[i])
        g_array.append(G)   
        
    # g_array contains layers from clear img to most blurry img
    return g_array

def laplacian_stack(gaussian_stack):
    
    g_stack = gaussian_stack.copy()
    l_stack = []
    # Create laplacian stack
    for i in range(0,len(g_stack)-1):
        G = cv2.pyrUp(g_stack[i+1])
        L = g_stack[i] - G
        l_stack.append(L)
    l_stack.append(g_stack[-1])
    
    # l_stack contains layers from finest high pass img, less finest high pass img,..., to blurry img
    return l_stack

def blend_img(img1,img2,mask):
    
    # Returns lists of 3D images
    l_img = laplacian_stack(gaussian_stack(img1))
    r_img = laplacian_stack(gaussian_stack(img2))
    g_mask = gaussian_stack(mask)
      
    blended_layers = [np.zeros_like(img1)] * len(r_img)
    
    # Element-wise multiplication and addition
    for layer in range(0, len(r_img)):
        left = np.multiply(g_mask[layer], l_img[layer])
        # Element-wise when subtracting an array from an integer
        right = np.multiply(1-g_mask[layer], r_img[layer])
        blended_layers[layer] = left + right
    
    return reconstruct_img(blended_layers)

def reconstruct_img(stack):
    for i in range(len(stack)):
        print(f"STACK {i}:", stack[i].shape)
    ls_ = stack[-1]
    for i in range(len(stack)-2, 1, -1):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, stack[i])
    # Should be axis 0 
    #reconstructed = np.sum(stack, axis=0)
        
    return (ls_ * 255).clip(0, 255).astype(np.uint8)
        
def img_show(label, img):
    cv2.namedWindow(label, cv2.WINDOW_NORMAL)
    cv2.imshow(label, img)
    
def main():
    # Declared variables for efficiency
    filename1 = "Naragdao_lab03_left"
    filename2 = "Naragdao_lab03_right"
    maskname1 = "mask"
    filetype1 = "png"
    filetype2 = "png"
    masktype1 = "png"
    #size = 512
    
    filename3 = "Naragdao_lab03_crazyone"
    filename4 = "Naragdao_lab03_crazytwo"
    maskname2 = "shirtmask"
    filetype3 = "png"
    filetype4 = "png"
    masktype2 = "png"
    #size = 512
    
    # Fetch images
    img1 = img_read(filename1, filetype1)
    img2 = img_read(filename2, filetype2)
    mask1 = img_read(maskname1, masktype1)
    
    img3 = img_read(filename3, filetype3)
    img4 = img_read(filename4, filetype4)
    mask2 = img_read(maskname2, masktype2)
    
    """ img1 = cv2.resize(img1, (size, size))
    img2 = cv2.resize(img2, (size, size))
    mask = cv2.resize(mask, (size, size)) """
    
    #Normalize
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        mask1 = mask1.astype(np.float32) / 255.0
        img3 = img3.astype(np.float32) / 255.0
        img4 = img4.astype(np.float32) / 255.0
        mask2 = mask2.astype(np.float32) / 255.0

    # If None return error else do blending
    if img1 is None:
        return "Error in reading photos"
    else:
       
        blendedvert = test_blending(img1,img2,mask1)
        
        # Wait for any user input to close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Output image as png
        cv2.imwrite(f'output/Naragdao_lab03_blendvert.png', blendedvert)
            
        blendcrazy = test_blending(img3,img4,mask2)
        
        # Wait for any user input to close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Output image as png
        cv2.imwrite(f'output/Naragdao_lab03_blendcrazy.png', blendcrazy)
    
def img_read(filename, type):
    return cv2.imread(f'input/{filename}.{type}')

def test_recostruction(img1):
    result = reconstruct_img(laplacian_stack(gaussian_stack(img1)))
    img_show("Original", img1) 
    img_show("Reconstruction", result)

def test_blending(img1,img2,mask):
    size = 1024
    img1 = cv2.resize(img1, (size, size))
    img2 = cv2.resize(img2, (size, size))
    mask = cv2.resize(mask, (size, size))
    blended = blend_img(img1, img2, mask)
    img_show("Original 1", img1)
    img_show("Original 2", img2)
    img_show("Blended", blended)
    
    return blended
    
def test_stacks(img):
    gaussian = gaussian_stack(img)
    laplacian = laplacian_stack(gaussian)
        
    for i in range(0,len(laplacian)):
        img_show(f"Laplacian {i}", laplacian[i])
        
    for i in range(0,len(gaussian)):
        img_show(f"Gaussian {i}", gaussian[i])

if __name__ == "__main__":
    main()