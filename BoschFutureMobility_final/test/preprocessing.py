import time

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

def preprocessing(frame, alpha, beta, gamma):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast_frame = cv2.convertScaleAbs(grayscale_frame, alpha=alpha, beta=beta)
    bilateral_frame = cv2.bilateralFilter(contrast_frame, 9, 10, 10)
    return bilateral_frame

while True:

    start = time.time()

    # ROI
    x_cv_ROI = 320
    frame_ROI = frame[x_cv_ROI:, :]

    preprocessed_frame_ROI = preprocessing(frame_ROI, alpha=1.8, beta=0, gamma=0)

    cv2.imshow("Frame", frame)
    cv2.imshow("Final Frame", preprocessed_frame_ROI)

    end = time.time()

    print("time = {}".format(end - start))

    cv2.waitKey(1)
    _, frame = cap.read()
