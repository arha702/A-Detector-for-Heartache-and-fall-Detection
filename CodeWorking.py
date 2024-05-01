import cv2
import mediapipe as mp

# Import libraries for message notification (choose one)
# Option 1: Plays a system sound (Windows only)
# import winsound

# Option 2: Displays a message on the screen using pyautogui
import pyautogui

# Initialize drawing utils and pose solution
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define shoulder and wrist landmark IDs
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST


# Define a function to calculate distance between landmarks
def calculate_distance(landmark1, landmark2, results):
    # Get landmark coordinates
    x1, y1 = results.pose_landmarks.landmark[landmark1].x, results.pose_landmarks.landmark[landmark1].y
    x2, y2 = results.pose_landmarks.landmark[landmark2].x, results.pose_landmarks.landmark[landmark2].y

    # Calculate distance using Euclidean formula
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance


# Define a function to process video frame
def process_video_frame(image):
    # Convert BGR image to RGB for MediaPipe processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
     results = pose.process(image)

    # Draw pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if both hands are on top of left shoulder
        left_wrist_distance = calculate_distance(LEFT_WRIST, LEFT_SHOULDER, results)
        right_wrist_distance = calculate_distance(RIGHT_WRIST, LEFT_SHOULDER, results)

        if left_wrist_distance < right_wrist_distance:
            # Check if left wrist is closer than a threshold (adjust as needed)
            if left_wrist_distance < 0.2:
                # Display message (choose one method)
                print("Both hands are on top of your left shoulder!")

                # Option 1: Play system sound (Windows only)
                # winsound.Beep(2500, 1000)  # Play a beep sound at 2500 Hz for 1 second

                # Option 2: Display message on screen using pyautogui
                pyautogui.alert("Both hands are on top of your left shoulder!", title="Pose Detection")

    # Display the processed image
    cv2.imshow('MediaPipe Pose Detection', image)


# Capture video from webcam
cap = cv2.VideoCapture(0)

# Process video frames until 'q' key is pressed
while True:
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading image fails, exit loop
        break

    process_video_frame(image)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
