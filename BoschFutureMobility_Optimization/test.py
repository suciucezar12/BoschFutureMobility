import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while True:

    frame = frame[210:, :]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    frame_canny = cv2.Canny(frame_blurred, 180, 255)
    cv2.imshow("Canny", frame_canny)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    _, frame = cap.read()
