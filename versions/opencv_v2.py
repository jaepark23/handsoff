import cv2
import time
import mediapipe as mp
import numpy as np
import time
import math
import os

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def hand_result_extraction(result, img):
    temp_keypoints = []
    if len(result.handedness) > 0:
        for keypoint in result.hand_landmarks:
            for coordinate in keypoint:
                height, width, channels = img.shape
                x_px = min(math.floor(coordinate.x * width), width - 1)
                y_px = min(math.floor(coordinate.y * height), height - 1)
                temp_keypoints.append((x_px, y_px))
        return temp_keypoints

cap = cv2.VideoCapture(0)

def run_loop():
    while True:
        coordinates = []
        hand_keypoints = []
        success, img = cap.read()
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        frame = np.array(img)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks: 
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_keypoints.append((cx, cy))

        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        if len(face):
            x1, y1, x2, y2 = face[0][0], face[0][1], face[0][0] + face[0][2], face[0][1] + face[0][3]
            for keypoint in hand_keypoints:
                if keypoint[0] <= x2 and keypoint[0] >= x1 and keypoint[1] <= y2 and keypoint[1] >= y1:
                    os.system('say "get the fuck off."')
                    # os.system('afplay {file directory}')
                    break

        # time.sleep(1)
