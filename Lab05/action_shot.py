import cv2
import glob

from stitching import cv2_stitch

#  Collector from lab05.py, pasted here to avoid circular dependencies for now
def fetch_images(directory, file_type):
    img_paths = glob.glob(f"{directory}/*.{file_type}")
    imgs = []
    
    for img in img_paths:
        img_read = cv2.imread(img)
        imgs.append(img_read)
        
    return imgs

#   Generic cv2 function to extract frames from video
def extract_frames(video_path, output_folder, frame_interval):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            cv2.imwrite(f"{output_folder}/frame_{frame_count}.jpg", frame)

        frame_count += 1

    cap.release()

#   Function to create action shot with background subtraction
def create_action_shot(frames_folder):
    #   Load frames
    frames = fetch_images(frames_folder, "jpg")
    action_frames = []
    
    #   Create background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    #   Apply background subtraction to each frame
    for frame in frames:
        fg_mask = background_subtractor.apply(frame)
        action_frames.append(cv2.bitwise_and(frame, frame, mask=fg_mask))

    #   Combine frames into action shot
    action_shot = cv2.hconcat(action_frames)    ##  TODO: Instead of horizontally stitching frames, overlay them on top of each other.

    return action_shot

video_path = "data/spike.mp4"
output_folder = "output"

frame_interval = 5
extract_frames(video_path, output_folder, frame_interval)

action_shot = create_action_shot(output_folder)

cv2.imwrite("action_shot.jpg", action_shot)
