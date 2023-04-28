import cv2 as cv
import mediapipe as mp

mpfacedetection=mp.solutions.face_detection
mpdraw=mp.solutions.drawing_utils
facedetection=mpfacedetection.FaceDetection(1.5)

cap=cv.VideoCapture(0)

while True:
    process,img=cap.read()
    imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = facedetection.process(imgrgb)

    if result.detections:
        for detection in result.detections:

            mpdraw.draw_detection(img, detection)

    cv.imshow('image',img)
    cv.waitKey(1)