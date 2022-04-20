import time

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

def preprocessing(frame, alpha, beta, gamma):
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", grayscale_frame)
    contrast_frame = cv2.convertScaleAbs(grayscale_frame, alpha=alpha, beta=beta)
    cv2.imshow("COntrast", contrast_frame)
    # bilateral_frame = cv2.bilateralFilter(contrast_frame, 9, 10, 10)
    # cv2.imshow("Bilateral", bilateral_frame)
    # gx_sobel = cv2.Sobel(bilateral_frame,  0, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    canny_frame = cv2.Canny(contrast_frame, 100, 150)
    return canny_frame

while True:

    start = time.time()

    # ROI
    x_cv_ROI = 320
    frame_ROI = frame[x_cv_ROI:, :]

    preprocessed_frame_ROI = preprocessing(frame_ROI, alpha=1.9, beta=0, gamma=0)

    cv2.imshow("Frame", frame)
    cv2.imshow("Final Frame", preprocessed_frame_ROI)

    end = time.time()

    print("time = {}".format(end - start))

    cv2.waitKey(1)
    _, frame = cap.read()
