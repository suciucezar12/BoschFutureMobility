import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, rgb = cap.read()

height = rgb.shape[0]
width = rgb.shape[1]

while True:

    # Conversion of color spaces --------------------------------
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_BGRA2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(rgb, cv2.COLOR_BGR2HLS)

    # extracting channels ---------------------------------------
    # _, l, _ = cv2.split(hls)
    # _, _, v = cv2.split(hsv)
    # (b, g, r) = cv2.split(rgb)

    # choose our ROI
    x_cv_ROI = 320
    cv2.line(rgb, (0, x_cv_ROI), (width, x_cv_ROI), (0, 0, 255), 3)

    # test filters ------------------------------------
    gaussian = cv2.GaussianBlur(grayscale, (5, 5), sigmaX=0)
    bilateral = cv2.bilateralFilter(grayscale, 9, 10, 15)

    # increase contrast
    alpha = 1.8
    beta = 0
    alpha_beta_image = cv2.convertScaleAbs(grayscale, alpha=alpha, beta=beta)

    # gamma correction
    gamma = 1.2
    table = np.array([((i / 255) ** gamma) * 255 for i in range(256)], np.uint8)
    gamma_image = cv2.LUT(grayscale, table)

    # alpha_beta + gamma
    alpha = 1.8
    beta = 0
    alpha_beta_image1 = cv2.convertScaleAbs(grayscale, alpha=alpha, beta=beta)
    gamma = 1.8
    table = np.array([((i / 255) ** gamma) * 255 for i in range(256)], np.uint8)
    alph_beta_gamma_image = cv2.LUT(alpha_beta_image1, table)

    # histogram equalization
    hist_eq = cv2.equalizeHist(grayscale)


    # Display ---------------------------------------------------
    # cv2.imshow("RGB", rgb)

    cv2.imshow("Grayscale", grayscale)
    # cv2.imshow("L_HLS", l)
    # cv2.imshow("V_HSV", v)

    # cv2.imshow("R", r)
    # cv2.imshow("G", g)
    # cv2.imshow("B", b)

    # cv2.imshow("Gaussian", gaussian)
    # cv2.imshow("Bilateral", bilateral)

    # cv2.imshow("Gamma correction", gamma_image)
    # cv2.imshow("Alpha_Beta image", alpha_beta_image)
    # cv2.imshow("Alpha_Beta_Gamma image", alph_beta_gamma_image)

    cv2.imshow("Histogram equalization", hist_eq)

    cv2.waitKey(1)

    _, rgb = cap.read()
