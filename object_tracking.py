"""

MAIN PYTHON SCRIPT
object_tracking.py
Homework 10: Object detection and tracking using image keypoints and descriptors.

Authors:  Ana Bárbara Quintero 544073
Organisation: UDEM
Due date: 6th May 2024

EXAMPLE TERMINAL CODE: 
$ python object_tracking.py --img_obj truck.png --video vid_sequence.mp4

MY TERMINAL CODE:
python object_tracking.py  --img_obj "/Users/anabi/Documents/GitHub/HW10_ObjectDetection/images/first_image.png"   --video "/Users/anabi/Documents/GitHub/HW10_ObjectDetection/images/IMG_1256.MOV"

"""
# Importing the necessary libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import libraries.ORB_detection as orb
import libraries.video as vid 
import libraries.count as cnt 

def parse_args():
    """
    Parse command line arguments for image paths.
    
    Returns:
        Namespace: Parsed command line arguments with paths to the images.
    """
    parser = argparse.ArgumentParser(description='Open a video file and display its frames')
    parser.add_argument('--img_obj', required=True, help='Path to the image file for object detection')
    parser.add_argument('--video', required=True, help='Path to the video file')
    args = parser.parse_args()
    return args

def run_pipeline(video_path, img_obj_path):

    video = vid.open_video(video_path)
    image = orb.load_image(img_obj_path)
       ##INPUTS: IMAGE. OUTPUTS:  orbs, keypoints, descriptors.
    orbs, ref_keyp, ref_desc = orb.detect_features(image)



    #INITIALIZE VARIABLES

    prev_position = None
    crossing_left_to_right = 0
    crossing_right_to_left = 0


    while True:
        frame = vid.read_frame(video)
        if frame is None:
            break

        ##AQUI DIBUJAMOS LAS LINEAS Y EL TEXTO
        vertical_frame = cnt.draw_vertical_line(frame.copy())
        mid_line_x = frame.shape[1] // 2

        frame_keyp, frame_desc = orb.detect_features(vertical_frame)
        matches = orb.match_features(ref_desc, frame_desc)


        # Update the crossing count
        if prev_position is not None:
            crossing = cnt.count_crossings(prev_position, circle_position, mid_line_x)
            if crossing == 1:
                crossing_left_to_right += 1
            elif crossing == -1:
                crossing_right_to_left += 1

        # Store the current position for the next iteration
        prev_position = circle_position

          # Add text overlays
        cv.line(frame, (0, mid_line_x), (frame.shape[1], mid_line_x), (0, 255, 0), 2)  # Draw mid-line
#        cv.circle(frame, (circle_position, 200), 50, (0, 0, 255), -1)  # Draw circle
        cv.putText(frame, f"Crossings L to R: {crossing_left_to_right}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(frame, f"Crossings R to L: {crossing_right_to_left}", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow("Frame", frame)

        



    return 

def load_and_resize_image(path, scale=0.4):
    """
    Load an image from a file and resize it.
    
    Args:
        path (str): Path to the image file.
        scale (float): Scaling factor for resizing the image.

    Returns:
        np.ndarray: The resized image.
    """
    image = cv.imread('/Users/anabi/Documents/GitHub/HW10_ObjectDetection/IMG_1256.MOV') # Load the image
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.") # Raise an error if the image is not found
    resized_image = cv.resize(image, (0, 0), fx=scale, fy=scale) # Resize the image
    return resized_image


def display_images(images):
    """
    Display multiple images in separate windows.

    Args:
        images (dict): Dictionary of window names and image data.
    """
    for window_name, image in images.items(): # Display each image in a separate window
        cv.imshow(window_name, image) 
    while True:
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

if __name__ == '__main__':
    """
    args = parse_args() # Parse command line arguments
    image1 = load_and_resize_image(args.image1) # Load and resize the image
    image2 = load_and_resize_image(args.image2)
    keypoints1, descriptors1 = detect_features(image1) # Detect features in the images
    keypoints2, descriptors2 = detect_features(image2)
    good_matches = match_features(descriptors1, descriptors2) # Match features
    matched_image = draw_matches(image1, keypoints1, image2, keypoints2, good_matches) 
    display_images({'image1': image1, 'image2': image2, 'Matches': matched_image}) # Display the images
"""
    

    video_path = parse_args().video  # Get the path to the video file from command-line arguments
    img_obj_path = parse_args().img_obj  # Get the path to the image file for object detection

    cap = vid.open_video(video_path)  # Open the video file
    if cap is not None:
        while True:
            frame = vid.read_frame(cap)  # Read a frame from the video
            if frame is None:
                break  # If frame couldn't be captured, exit the loop
            cv.imshow('Frame', frame)  # Display the frame
            if cv.waitKey(25) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
                break
        vid.close_video(cap)  # Close the video file and destroy all windows
