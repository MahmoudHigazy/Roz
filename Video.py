import numpy as np
import cv2
from roz import *


def MyKernel():
    global CroppedImage
    AttendanceDone=False
    CroppedImage = []
    cap = cv2.VideoCapture(0)
    MyTemplate = cap.read()
    FrameNumber = 0
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    xhTemp = 0
    yhTemp = 0
    NFrames = 10
    Counter=0
    while (True):
        ret, frame = cap.read()
        Image = frame
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detect_image(frame, True)
        xh, yh, wh, hh = 0, 0, 0, 0
        for f in face:
            xh = f.left()
            yh = f.top()
            wh = f.right()
            hh = f.bottom()

            Image = cv2.rectangle(frame, (xh, yh), (wh, hh), (255, 0, 0), 2)

            if abs(xh - xhTemp) < 10:
                Counter = Counter + 1
            else:
                Counter = 0
            if Counter == NFrames:
                # AttendanceDone=True
                cv2.imwrite('test.jpg', frame)
                return frame
            xhTemp = xh
            yhTemp = yh

        cv2.imshow('frame', Image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print{FrameNumber}
            break
        FrameNumber = FrameNumber + 1

    cap.release()
    cv2.destroyAllWindows()
    # cv2.imshow('frame', CroppedImage)
    # cv2.waitKey(0)
    return None

Image=MyKernel()