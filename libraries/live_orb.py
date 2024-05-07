"""
orb_live.py
This script is used to detect the keypoints and descriptors of an image using the ORB algorithm, and compare it in real time with a live camera feed.

Authors: Alberto Castro Villasana, Ana Bárbara Quintero, Héctor Camacho Zamora
Organisation: UDEM
First created on Friday 23 April 2024

python live_orb.py --image1 "/Users/anabi/Documents/GitHub/HW10_ObjectDetection/IMG_1259.JPG"

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
    image = cv2.imread(path)  # Load the image
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")  # Raise an error if the image is not found
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)  # Resize the image
    return resized_image

def detect_features(image):
    """
    Detect and compute ORB features and descriptors in the image.
    
    Args:
        image (np.ndarray): Image in which to detect features.

    Returns:
        tuple: Keypoints and descriptors of the image.
    """
    orb = cv2.ORB_create(nfeatures=1000)  # Create an ORB object, nfetures is the maximum number of features to retain
    keypoints, descriptors = orb.detectAndCompute(image, None)  # Detect keypoints and compute descriptors
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

def run_pipeline():
    """
    Run the ORB feature matching pipeline with live camera input.
    """
    args = parse_args()  # Parse command line arguments
    image1 = load_and_resize_image(args.image1)  # Load and resize the reference image
    keypoints1, descriptors1 = detect_features(image1)  # Detect features in the reference image
    
    cap = cv2.VideoCapture(0)  # Open the first camera connected to the system
    
    while True:
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            continue  # Skip the rest of the loop if frame couldn't be captured

        keypoints2, descriptors2 = detect_features(frame)  # Detect features in the current frame
        good_matches = match_features(descriptors1, descriptors2)  # Match features
        matched_image = cv2.drawMatchesKnn(image1, keypoints1, frame, keypoints2, good_matches, None, flags=2)
        
        cv2.imshow('Matches', matched_image)  # Display the image with matches
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_pipeline()  # Run the pipeline