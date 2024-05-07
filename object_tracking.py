
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
python object_tracking.py --img_obj "/Users/anabi/Documents/GitHub/HW10_ObjectDetection/images/first_image.png" --video "/Users/anabi/Documents/GitHub/HW10_ObjectDetection/images/IMG_1256.MOV"


"""

# Importing the necessary libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import libraries.ORB_detection as orb
import libraries.video as vid 

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

def detect_features(image):
    """
    Detect and compute ORB features and descriptors in the image.
    
    Args:
        image (np.ndarray): Image in which to detect features.

    Returns:
        tuple: Keypoints and descriptors of the image.
    """
    orb = cv.ORB_create(nfeatures=1000) # Create an ORB object, nfetures is the maximum number of features to retain
    keypoints, descriptors = orb.detectAndCompute(image, None) # Detect keypoints and compute descriptors
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    Match ORB features using the Brute Force matcher.

    Args:
        desc1 (np.ndarray): Descriptors of the first image.
        desc2 (np.ndarray): Descriptors of the second image.

    Returns:
        list: Good matches after applying ratio test.
    """
    bf = cv.BFMatcher() # Create a Brute Force matcher
    matches = bf.knnMatch(desc1, desc2, k=2) # Match descriptors of the two images
    good_matches = [] # List to store good matches
    for m, n in matches: # Apply ratio test
        if m.distance < 0.5 * n.distance: # If the distance is less than 0.5 times the next closest distance
            good_matches.append([m])
    return good_matches

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    """
    Draw matches between two images.

    Args:
        image1 (np.ndarray): First image.
        keypoints1 (list): Keypoints in the first image.
        image2 (np.ndarray): Second image.
        keypoints2 (list): Keypoints in the second image.
        matches (list): Good matches to draw.

    Returns:
        np.ndarray: Image with drawn matches.
    """
    return cv.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None, flags=2) # Draw matches

def display_images(images):
    """
    Display multiple images in separate windows.

    Args:
        images (dict): Dictionary of window names and image data.
    """
    for window_name, image in images.items(): # Display each image in a separate window
        cv.imshow(window_name, image) 
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
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

    video_path = 'path/to/your/video/file.mp4'  # Provide the path to your video file
    cap = vid.open_video(video_path)  # Open the video file
    if cap is not None:
        while True:
            frame = vid.read_frame(cap)  # Read a frame from the video
            if frame is None:
                break  # If frame couldn't be captured, exit the loop
            cv.imshow('Frame', frame)  # Display the frame
            if cv.waitKey(25) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
                break
        vid.close_video(cap)  # Close the video file and destroy all windows"""
    

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
