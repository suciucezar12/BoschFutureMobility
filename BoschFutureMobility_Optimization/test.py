import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while True:

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

    _, frame = cap.read()
