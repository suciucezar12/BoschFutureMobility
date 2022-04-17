import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while True:

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("Frame", frame)
    cv2.imshow("HSV", hsv)
    cv2.waitKey(1)

    _, frame = cap.read()
