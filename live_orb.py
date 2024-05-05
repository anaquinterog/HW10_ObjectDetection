"""
orb_live.py
This script is used to detect the keypoints and descriptors of an image using the ORB algorithm, and compare it in real time with a live camera feed.

Authors: Alberto Castro Villasana, Ana Bárbara Quintero, Héctor Camacho Zamora
Organisation: UDEM
First created on Friday 23 April 2024
"""

# Importing the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse_args():
    """
    Parse command line arguments for image path.
    
    Returns:
        Namespace: Parsed command line arguments with path to the reference image.
    """
    parser = argparse.ArgumentParser(description='ORB feature matching between a reference image and live camera feed.')
    parser.add_argument('--image1', type=str, help='Path to the reference image.')
    return parser.parse_args()

def load_and_resize_image(path, scale=0.4):
    """
    Load an image from a file and resize it.
    
    Args:
        path (str): Path to the image file.
        scale (float): Scaling factor for resizing the image.

    Returns:
        np.ndarray: The resized image.
    """
    image1 = cv2.imread('/Users/anabi/Documents/GitHub/HW10_ObjectDetection/images/Screen Shot 2024-05-05 at 1.12.15 PM.png')  # Load the image
    if image1 is None:
        raise FileNotFoundError(f"Image at path {path} not found.")  # Raise an error if the image is not found
    image1 = cv2.resize(image1, (0, 0), fx=scale, fy=scale)  # Resize the image
    return image1

def detect_features(image1):
    """
    Detect and compute ORB features and descriptors in the image.
    
    Args:
        image (np.ndarray): Image in which to detect features.

    Returns:
        tuple: Keypoints and descriptors of the image.
    """
    orb = cv2.ORB_create(nfeatures=1000)  # Create an ORB object, nfetures is the maximum number of features to retain
    keypoints, descriptors = orb.detectAndCompute(image1, None)  # Detect keypoints and compute descriptors
    return keypoints, descriptors

def match_features(desc1, desc2):
    """
    Match ORB features using the Brute Force matcher.

    Args:
        desc1 (np.ndarray): Descriptors of the first image.
        desc2 (np.ndarray): Descriptors of the camera feed.

    Returns:
        list: Good matches after applying ratio test.
    """
    bf = cv2.BFMatcher()  # Create a Brute Force matcher
    matches = bf.knnMatch(desc1, desc2, k=2)  # Match descriptors
    good_matches = []  # List to store good matches
    for m, n in matches:  # Apply ratio test
        if m.distance < 0.7 * n.distance:  # If the distance is less than 0.7 times the next closest distance
            good_matches.append([m])
    return good_matches
import cv2

def run_pipeline(video_path):
    """
    Run the ORB feature matching pipeline with video input.

    The pipeline consists of the following steps:
    1. Parse command line arguments for the path to the reference image.
    2. Load and resize the reference image.
    3. Detect ORB features and compute descriptors in the reference image.
    4. Open the video file.
    5. In an infinite loop, read a frame from the video and:
        - Detect ORB features and compute descriptors in the current frame.
        - Match features between the reference image and the current frame using Brute Force matching with a ratio test.
        - Draw matches between the reference image and the current frame.
        - Display the image with matches.
        - Exit the loop if 'q' is pressed.
    6. Close the video file and destroy all windows.
    """
    args = parse_args()  # Parse command line arguments
    image1 = load_and_resize_image(args.image1)  # Load and resize the reference image
    keypoints1, descriptors1 = detect_features(image1)  # Detect features in the reference image
    
    cap = open_video(video_path)  # Open the video file
    
    while True:
        frame = read_frame(cap)  # Read a frame from the video
        if frame is None:
            break  # If frame couldn't be captured, exit the loop
        
        keypoints2, descriptors2 = detect_features(frame)  # Detect features in the current frame
        good_matches = match_features(descriptors1, descriptors2)  # Match features
        matched_image = cv2.drawMatchesKnn(image1, keypoints1, frame, keypoints2, good_matches, None, flags=2)
        
        show_frame('Matches', matched_image)  # Display the image with matches
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
            break
    
    close_video(cap)  # Close the video file and destroy all windows


def open_video(video_path):
    """ Open the video file from the given path. """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return None
    return cap


def read_frame(cap):
    """ Read a frame from the video and display it. """
    ret, frame = cap.read()
    if ret:
        #cv2.imshow(window_name, frame)
        return frame
    return None

def show_frame(window_name, frame):
    """ Show the frame in a window. """
    cv2.imshow(window_name, frame)

def close_video(cap):
    """ Close the video file and destroy all windows. """
    cap.release()
    cv2.destroyAllWindows()




def parse_args():
    """
    Parse command line arguments for image path.
    
    Returns:
        Namespace: Parsed command line arguments with path to the reference image.
    """
    parser = argparse.ArgumentParser(description='ORB feature matching between a reference image and live camera feed.')
    parser.add_argument('--image1', type=str, required=True, help='Path to the reference image.')
    args = parser.parse_args()
    
    if not args.image1:
        raise ValueError("--image1 argument is required.")
    
    return args

def load_and_resize_image(image_path):
    # Add your image loading and resizing logic here
    pass

def detect_features(image):
    # Add your feature detection logic here
    pass

def match_features(descriptors1, descriptors2):
    # Add your feature matching logic here
    pass

if __name__ == '__main__':
    video_path = '/Users/anabi/Documents/GitHub/HW10_ObjectDetection/images/IMG_1256.MOV'  # Specify the path to your video file
    run_pipeline(video_path)  # Run the pipeline with the specified video path

#python -u "/Users/anabi/Documents/GitHub/HW10_ObjectDetection/live_orb.py" --image1 "/Users/anabi/Documents/GitHub/HW10_ObjectDetection/images/Screen Shot 2024-05-05 at 1.12.15 PM.png"