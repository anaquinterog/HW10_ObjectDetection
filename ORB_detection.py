import cv2 as cv
import numpy as np

def load_image(image_path):
    """Load the reference image."""
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print("Error loading image")
        return None
    return image


def detect_features(image):
    """
    Detect and compute ORB features and descriptors in the image.
    
    Args:
        image (np.ndarray): Image in which to detect features.

    Returns:
        tuple: Keypoints and descriptors of the image.
    """
    orb = cv.ORB_create(nfeatures=1000)  # Create an ORB object, nfetures is the maximum number of features to retain
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
    bf = cv.BFMatcher()  # Create a Brute Force matcher
    matches = bf.knnMatch(desc1, desc2, k=2)  # Match descriptors
    good_matches = []  # List to store good matches
    for m, n in matches:  # Apply ratio test
        if m.distance < 0.7 * n.distance:  # If the distance is less than 0.7 times the next closest distance
            good_matches.append([m])
    return good_matches
