import cv2
import time
import mediapipe as mp
import numpy as np
import time
import math

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

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
with HandLandmarker.create_from_options(hand_options) as landmarker:
    while True:
        coordinates = []
        hand_keypoints = []
        success, img = cap.read()
        frame = np.array(img)
        timestamp = int(round(time.time()*1000))
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # corner1 = (face[0][0], face[0][1])
        # corner2 = (face[0][0] + face[0][2], face[0][1] + face[0][3])

        hand_detector_result = landmarker.detect(mp_image)
        try:
            hand_keypoints = hand_keypoints + hand_result_extraction(hand_detector_result, img)
        except:
            pass

        # for keypoint in hand_keypoints:
        #     height, width, channels = img.shape
        #     x_px = min(math.floor(keypoint[0] * width), width - 1)
        #     y_px = min(math.floor(keypoint[1] * height), height - 1)
        #     cv2.circle(img, (x_px, y_px), 2, (0, 255, 0), 2)
    
        if len(face):
            x1, y1, x2, y2 = face[0][0], face[0][1], face[0][0] + face[0][2], face[0][1] + face[0][3]
            for keypoint in hand_keypoints:
                # print(keypoint)
                # print(x1, y1, x2, y2)
                if keypoint[0] <= x2 and keypoint[0] >= x1 and keypoint[1] <= y2 and keypoint[1] >= y1:
                    print('gtfo')
                else:
                    print('oops')

        for (x, y, w, h) in face:
            # print((x, y, w, h))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("GTFO", img_rgb)
        
        cv2.waitKey(1)

        if cv2.waitKey(1) == ord('q'):
            break
            # When everything done, release the capture
