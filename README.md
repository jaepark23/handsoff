## What is GTFO
- Program that detects when a user touches their face with their hands
## How does it work
- Utilizes hand and face detection models to track coordinates and check when hand and face coordinates overlap each other
- The program will flag when the hand coordinates overlap the face coordinates
- Uses your main webcam to track your hand and face 
## Why
- To help prevent acne by alerting when users touch their face
## Tools used
- Python
- MediaPipe hand detection model
- OpenCV face detection model
