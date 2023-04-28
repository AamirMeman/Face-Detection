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

            boxl=detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox=int(boxl.xmin *iw),int(boxl.ymin *ih),\
                int(boxl.width * iw), int(boxl.height * ih)
            cv.rectangle(img,bbox,(255,0,255),2)

            cv.putText(img,f'{int(detection.score[0] * 100)}%',
                       (bbox[0],bbox[1]-20),cv.FONT_HERSHEY_PLAIN,2
                       ,(0,0,255),2)
    cv.imshow('image',img)
    cv.waitKey(1)