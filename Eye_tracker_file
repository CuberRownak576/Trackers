import cv2
import mediapipe as mp
import pyautogui
import time
import queue
import threading
from pynput.mouse import Controller

# Initialize camera and mouse
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
mouse = Controller()

# Queue for frame capture
frame_queue = queue.Queue()


# Frame capture thread
def capture_video():
    while True:
        ret, frame = cam.read()
        if ret:
            frame_queue.put(frame)


capture_thread = threading.Thread(target=capture_video, daemon=True)
capture_thread.start()

# Constants
SPEED_FACTOR = 1.58
frame_skip = 3
frame_counter = 0

while True:
    frame = frame_queue.get()  # Get frame from the queue
    frame_counter += 1

    if frame_counter % frame_skip == 0:
        frame_h, frame_w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(frame_rgb)
        landmarks_points = output.multi_face_landmarks

        if landmarks_points:
            landmarks = landmarks_points[0].landmark
            left_eye = [landmarks[145], landmarks[159]]  # Left eye landmarks for blink detection
            for landmark in left_eye:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))  # Yellow circles for left eye

            # Blink detection and clicking
            if (left_eye[0].y - left_eye[1].y) < 0.01:  # Detect eye blink
                pyautogui.click()
                time.sleep(1)  # Prevent double-clicking

            # Cursor movement based on left eye position (use only the second left-eye landmark)
            x = int(left_eye[1].x * frame_w)
            y = int(left_eye[1].y * frame_h)
            screen_x = screen_w / frame_w * x * SPEED_FACTOR
            screen_y = screen_h / frame_h * y * SPEED_FACTOR
            mouse.position = (screen_x, screen_y)  # Move cursor instantly

        frame_counter = 0

    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)
