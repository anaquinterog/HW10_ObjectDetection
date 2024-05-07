import cv2 as cv

def open_video(video_path):
    """
    Open the video file from the given path.

    Args:
        video_path (str): Path to the video file.

    Returns:
        cv2.VideoCapture: VideoCapture object for the video if successful,
                          None otherwise.
    """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return None
    return cap

def read_frame(cap):
    """
    Read a frame from the video and return it.

    If the frame couldn't be read, return None.

    Args:
        cap (cv2.VideoCapture): VideoCapture object of the video.

    Returns:
        np.ndarray or None: Frame read from the video if successful,
                            None otherwise.
    """
    ret, frame = cap.read()  # Read a frame from the video
    if ret:  # If the frame was read successfully
        return frame  # Return the frame
    return None  # Otherwise return None

def close_video(cap):
    """
    Close the video file and destroy all windows.

    This function releases the video file and destroys all the windows created by
    the program. It should be called at the end of the program to prevent memory
    leaks.

    Args:
        cap (cv2.VideoCapture): VideoCapture object of the video.
    """
    cap.release()
    cv.destroyAllWindows()
