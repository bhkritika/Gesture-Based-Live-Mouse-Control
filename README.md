# Hand Gesture Control Using OpenCV and MediaPipe

This project enables controlling mouse cursor actions (move, left click, right click, double click, zoom in/out) using hand gestures via webcam input.

## 🔧 Technologies Used
- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- pynput

## ✋ Features
- Mouse movement using index finger
- Left click, right click, and double click gestures
- Zoom in/out using thumb-index distance
- Real-time gesture detection

## 📁 Files
- `show.py`: Main application script.
- `utils.py`: Utility functions like distance, angle, gesture detection.

## ▶️ How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-control.git
   cd hand-gesture-control

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   python show.py

## 📷 Gestures Supported
✋ Move Mouse
👆 Left Click
👉 Right Click
👇 Double Click
🤏 Zoom In/Out

## ⚠️ Requirements
- A webcam
- Works best in good lighting
