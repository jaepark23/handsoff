import cv2
import time
import mediapipe_img as mp
import numpy as np
import time
import math

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if len(result.handedness) > 0:
        for keypoint in result.hand_landmarks:
            for coordinate in keypoint:
                hand_keypoints.append((round(coordinate.x, 1), round(coordinate.y, 1)))
                # height, width, channels = img.shape
                # x_px = min(math.floor(coordinate.x * image_width), image_width - 1)
                # y_px = min(math.floor(coordinate.y * image_height), image_height - 1)
    return "test"

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

def print_result2(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    # for keypoint in result.detections[0].keypoints:
    #     print(keypoint)
    #     print('-----------------------')
    if len(result.detections) > 0:
        for keypoint in result.detections[0].keypoints:
            face_keypoints.append((round(keypoint.x, 1), round(keypoint.y, 1)))

face_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result2)

with HandLandmarker.create_from_options(hand_options) as landmarker, FaceDetector.create_from_options(face_options) as detector:
    cap = cv2.VideoCapture(0)
    while True:
        hand_keypoints = []
        face_keypoints = []
        success, img = cap.read()
        frame = np.array(img)
        timestamp = int(round(time.time()*1000))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        hand_result = landmarker.detect_async(mp_image, timestamp)
        face_result = detector.detect_async(mp_image, timestamp)

        # print(hand_keypoints)
        print(face_keypoints)
        # for keypoint in hand_keypoints:
        #       height, width, channels = img.shape
        #       x_px = min(math.floor(keypoint[0] * width), width - 1)
        #       y_px = min(math.floor(keypoint[1] * height), height - 1)
        #       cv2.circle(img, (x_px, y_px), 2, (0, 255, 0), 2)
        
        # for keypoint in face_keypoints:
        #       height, width, channels = img.shape
        #       x_px = min(math.floor(keypoint[0] * width), width - 1)
        #       y_px = min(math.floor(keypoint[1] * height), height - 1)
        #       cv2.circle(img, (x_px, y_px), 2, (0, 255, 0), 2)

        cv2.imshow("Test", img)
        cv2.waitKey(1)
        # if set(hand_keypoints) & set(face_keypoints):
        #     print("get tf off ")
        # else:
        #     print("oops")
        # if cv2.waitKey(1) == ord('q'):
        #     break
        #     # When everything done, release the capture