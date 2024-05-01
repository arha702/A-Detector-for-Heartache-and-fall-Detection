import time

import cv2
import mediapipe as mp
import math
import numpy as np
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
## Setup mediapipe instance

LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST
def calculate_distance(landmark1, landmark2, results):
    # Get landmark coordinates
    x1, y1 = results.pose_landmarks.landmark[landmark1].x, results.pose_landmarks.landmark[landmark1].y
    x2, y2 = results.pose_landmarks.landmark[landmark2].x, results.pose_landmarks.landmark[landmark2].y

    # Calculate distance using Euclidean formula
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance

    start_time = None
    hands_on_shoulder = False
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        try:
            landmarks = results.pose_landmarks
            # print(landmarks)

            # shoulder_x = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x]
            # shoulder_y =[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            # left_pinkie_x = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x]
            # left_pinkie_y = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
            # right_pinkie_x =[landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x]
            # right_pinkie_y = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]

            left_wrist_distance = calculate_distance(LEFT_WRIST, LEFT_SHOULDER, results)
            right_wrist_distance = calculate_distance(RIGHT_WRIST, LEFT_SHOULDER, results)

            # Keep track of time spent with hands on shoulder


            if left_wrist_distance < right_wrist_distance:
                # Check if left wrist is closer than a threshold
                if left_wrist_distance < 0.2:
                    # Hands are on shoulder, start timer if not already started
                    print("Timer Started")
                    print(time.time())
                    if not hands_on_shoulder:
                        start_time = time.time()
                        hands_on_shoulder = True

                    # Check if hands have been on shoulder for at least 5 seconds
                    if time.time() - start_time >= 5:
                        # Display message (choose one method)
                        print("Both hands have been on your left shoulder for 5 seconds")
                        pyautogui.alert("Chest Pain Detected", title="Pose Detection")
                    # Option 2: Display message on screen using pyautogui
                        start_time = 0
                        hands_on_shoulder = False
                else:
                    # Hands are not on shoulder, reset timer
                    hands_on_shoulder = False
                    start_time = 0






        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




