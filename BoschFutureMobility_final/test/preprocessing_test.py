import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, rgb = cap.read()

while True:

    # Conversion of color spaces --------------------------------
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_BGRA2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(rgb, cv2.COLOR_BGR2HLS)

    # extracting channels ---------------------------------------
    # _, l, _ = cv2.split(hls)
    # _, _, v = cv2.split(hsv)
    # (b, g, r) = cv2.split(rgb)

    # test filters ------------------------------------
    gaussian = cv2.GaussianBlur(grayscale, (5, 5), sigmaX=0)

    # Display ---------------------------------------------------
    # cv2.imshow("RGB", rgb)

    cv2.imshow("Grayscale", grayscale)
    cv2.imshow("Gaussian", gaussian)
    # cv2.imshow("L_HLS", l)
    # cv2.imshow("V_HSV", v)

    # cv2.imshow("R", r)
    # cv2.imshow("G", g)
    # cv2.imshow("B", b)

    cv2.waitKey(1)

    _, rgb = cap.read()
