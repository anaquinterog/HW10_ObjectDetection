import cv2 as cv

def draw_vertical_line(frame):
    """
    Draw a vertical line down the middle of the frame.

    Args:
        frame (numpy.ndarray): Input frame.

    Returns:
        numpy.ndarray: Frame with a vertical line drawn down the middle.
    """
    height, width, _ = frame.shape
    line_position = width // 2  # Position of the vertical line
    
    # Draw the vertical line
    cv.line(frame, (line_position, 0), (line_position, height), (0, 255, 0), thickness=2)

    return frame


def vertical_line_cout(vertical_frame):
    draw_vertical_line(vertical_frame)