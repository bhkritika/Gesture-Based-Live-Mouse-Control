import math
import numpy as np

def get_angle(a, b, c):
    """Calculate the angle between three points."""
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    return np.abs(np.degrees(radians))

def get_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_finger_bent(landmark_list, finger_tip):
    """Check if a given finger is bent."""
    return landmark_list[finger_tip][1] > landmark_list[finger_tip - 2][1]  # Compare y-coordinates

def is_swipe_up(landmark_list):
    """Detect Swipe Up gesture."""
    if len(landmark_list) < 21:
        return False
    wrist = landmark_list[0]  # Wrist position
    index_tip = landmark_list[8]  # Index finger tip
    return wrist[1] > index_tip[1]  # Wrist moves up relative to index tip
