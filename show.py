import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
from pynput.mouse import Button, Controller

# Initialize mouse controller
mouse = Controller()
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Drawing utility
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Constants
ZOOM_IN_THRESHOLD = 35
ZOOM_OUT_THRESHOLD = -35
SMOOTHING_FACTOR = 0.2  # Smoothing for mouse movement

prev_thumb_index_dist = None
prev_mouse_x, prev_mouse_y = screen_width // 2, screen_height // 2  # Initialize to center of screen

def get_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_finger_bent(landmark_list, finger_tip):
    """Check if a given finger is bent."""
    return landmark_list[finger_tip][1] > landmark_list[finger_tip - 2][1]  # Compare y-coordinates

def smooth_cursor(x, y):
    """Smooth mouse movement using exponential moving average."""
    global prev_mouse_x, prev_mouse_y
    new_x = int(prev_mouse_x * (1 - SMOOTHING_FACTOR) + x * SMOOTHING_FACTOR)
    new_y = int(prev_mouse_y * (1 - SMOOTHING_FACTOR) + y * SMOOTHING_FACTOR)
    prev_mouse_x, prev_mouse_y = new_x, new_y
    return new_x, new_y

def move_mouse(index_finger_tip):
    """Move the mouse cursor smoothly based on the finger tip position."""
    if index_finger_tip:
        x = int(index_finger_tip[0] * screen_width)
        y = int(index_finger_tip[1] * screen_height)
        smoothed_x, smoothed_y = smooth_cursor(x, y)
        pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.05)

def detect_gestures(frame, landmark_list):
    """Detect and perform various hand gestures."""
    global prev_thumb_index_dist

    if len(landmark_list) < 21:
        return

    index_finger_tip = landmark_list[8]
    thumb_tip = landmark_list[4]
    thumb_index_dist = get_distance(thumb_tip, index_finger_tip)

    # Move Cursor
    move_mouse(index_finger_tip)

    # Left Click Gesture
    if is_finger_bent(landmark_list, 8) and not is_finger_bent(landmark_list, 12):
        mouse.press(Button.left)
        mouse.release(Button.left)
        cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Right Click Gesture
    elif is_finger_bent(landmark_list, 12) and not is_finger_bent(landmark_list, 8):
        mouse.press(Button.right)
        mouse.release(Button.right)
        cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Double Click Gesture
    elif is_finger_bent(landmark_list, 8) and is_finger_bent(landmark_list, 12):
        pyautogui.doubleClick()
        cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Zoom In / Out Gesture
    elif prev_thumb_index_dist is not None:
        if thumb_index_dist - prev_thumb_index_dist > ZOOM_IN_THRESHOLD:
            pyautogui.hotkey('ctrl', '+')
            cv2.putText(frame, "Zoom In", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif thumb_index_dist - prev_thumb_index_dist < ZOOM_OUT_THRESHOLD:
            pyautogui.hotkey('ctrl', '-')
            cv2.putText(frame, "Zoom Out", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    prev_thumb_index_dist = thumb_index_dist

def main():
    """Main function to capture hand gestures."""
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                for hand_landmarks in processed.multi_hand_landmarks:
                    # Draw landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Store (x, y) coordinates of all landmarks
                    landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

            detect_gestures(frame, landmark_list)

            cv2.imshow('Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
