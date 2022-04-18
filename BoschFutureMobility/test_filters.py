import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while True:

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    bgr_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    h, l, s = cv2.split(hls)

    # h, s, v = cv2.split(hsv)

    cv2.imshow("Frame", frame)
    # cv2.imshow("HSV", hsv)
    cv2.imshow("H", h)
    cv2.imshow("L", l)
    cv2.imshow("S", s)
    cv2.waitKey(1)

    _, frame = cap.read()
