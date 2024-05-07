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



def count_crossings(prev_position, current_position, mid_line):
    """
    Count the number of crossings between a circle and a vertical line.

    Parameters:
        prev_position (int): The previous position of the circle.
        current_position (int): The current position of the circle.
        mid_line (int): The x-coordinate of the vertical line.

    Returns:
        int: The number of crossings. Returns 1 if the circle crosses from left to right, -1 if it crosses from right to left, and 0 if it does not cross.
    """


    crossing = 0
    if prev_position < mid_line and current_position >= mid_line:
        crossing = 1  # Circle crosses from left to right
    elif prev_position > mid_line and current_position <= mid_line:
        crossing = -1  # Circle crosses from right to left
    return crossing
