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

while True:
    coordinates = []
    hand_keypoints = []
    success, img = cap.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: 
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_keypoints.append((cx, cy))

            # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    frame = np.array(img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    if len(face):
        x1, y1, x2, y2 = face[0][0], face[0][1], face[0][0] + face[0][2], face[0][1] + face[0][3]
        for keypoint in hand_keypoints:
            if keypoint[0] <= x2 and keypoint[0] >= x1 and keypoint[1] <= y2 and keypoint[1] >= y1:
                os.system('afplay soundeffect.mp3')

    # for (x, y, w, h) in face:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imshow("GTFO", img)
    
    cv2.waitKey(1)

    if cv2.waitKey(1) == ord('q'):
        break
        # When everything done, release the capture
