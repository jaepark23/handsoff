# mediapipe solution

import cv2
import time
import mediapipe as mp
import numpy as np
import time
import math

# model initialization
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

face_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

# data extraction functions
def hand_result_extraction(result):
    temp_keypoints = []
    if len(result.handedness) > 0:
        for keypoint in result.hand_landmarks:
            for coordinate in keypoint:
                temp_keypoints.append((round(coordinate.x, 1), round(coordinate.y, 1)))
        return temp_keypoints

def face_result_extraction(result):
    temp_keypoints = []
    if len(result.detections) > 0:
        for keypoint in result.detections[0].keypoints:
            temp_keypoints.append((round(keypoint.x, 1), round(keypoint.y, 1)))
    return temp_keypoints, result.detections[0].bounding_box

    # run inferencing
with FaceDetector.create_from_options(face_options) as detector, HandLandmarker.create_from_options(hand_options) as landmarker:
    cap = cv2.VideoCapture(0)
    while True:
        hand_keypoints = []
        face_keypoints = []
        success, img = cap.read()
        frame = np.array(img)
        timestamp = int(round(time.time()*1000))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_detector_result = landmarker.detect(mp_image)
        face_detector_result = detector.detect(mp_image)
        # print(hand_detector_result)
        # print(face_detector_result)
        try:
            hand_keypoints = hand_keypoints + hand_result_extraction(hand_detector_result)
        except:
            pass
        try:
            keypoints, box = face_result_extraction(face_detector_result)
            face_keypoints = face_keypoints + keypoints
            
        except: 
            pass

        for keypoint in hand_keypoints:
            height, width, channels = img.shape
            x_px = min(math.floor(keypoint[0] * width), width - 1)
            y_px = min(math.floor(keypoint[1] * height), height - 1)
            cv2.circle(img, (x_px, y_px), 2, (0, 255, 0), 2)
        

        # cv2.rectangle(img, (box.origin_x, box.origin_y), (box.origin_x + box.width, box.origin_y + box.height), (255, 0, 0), 2)
        for keypoint in face_keypoints:
            height, width, channels = img.shape
            x_px = min(math.floor(keypoint[0] * width), width - 1)
            y_px = min(math.floor(keypoint[1] * height), height - 1)
            cv2.circle(img, (x_px, y_px), 2, (0, 255, 0), 2)

        cv2.imshow("GTFO", img)
        
        cv2.waitKey(1)

        # if set(hand_keypoints) & set(face_keypoints):
        #     print("get tf off ")
        # else:
        #     print("oops")

        if cv2.waitKey(1) == ord('q'):
            break
            # When everything done, release the capture
