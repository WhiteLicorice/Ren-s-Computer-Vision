import cv2
import glob

def main():
    stitcher()

# Image stitching function using stitcher
def stitcher():
        # Load the images
    img_list = uploadAllImages()

    # Load Stitcher
    imgStitcher = cv2.Stitcher_create()
    
    # Image Stitching
    error, stitched_img = imgStitcher.stitch(img_list)
    
    # Saves stitched image if no error occured
    if not error:
        cv2.imwrite("data/output2.png", stitched_img)
        showImg("Stitched Image", stitched_img)

def uploadAllImages():
    
    # Gets all images in the path
    imgPaths = glob.glob("dataTest/*.jpg")
    imgs = []
    
    for img in imgPaths:
        imgRead = cv2.imread(img)
        imgs.append(imgRead)
        
    return imgs


def showImg(label, img):
    cv2.namedWindow(label, cv2.WINDOW_NORMAL)
    cv2.imshow(label, img)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    main()
    