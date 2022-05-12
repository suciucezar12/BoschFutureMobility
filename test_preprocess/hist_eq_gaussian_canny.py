import time

import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    start = time.time()

    frame_ROI = frame[270:, :]

    gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    hist_eq = cv2.equalizeHist(gray)
    cv2.imshow("Hist_eq", hist_eq)

    gaussian = cv2.GaussianBlur(hist_eq, ksize=(3, 3), sigmaX=0)
    # _, otsu = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny = cv2.Canny(hist_eq, 150, 250)

    end = time.time()

    print("time = {}".format(end - start))

    # cv2.imshow("Otsu", otsu)
    cv2.imshow("Canny", canny)
    cv2.waitKey(1)
