import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while True:

    frame = frame[210:, :]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blurred = cv2.GaussianBlur(frame_gray, (9, 9), 0)
    frame_canny = cv2.Canny(frame_blurred, 180, 255)

    lines_candidate = cv2.HoughLinesP(frame_canny, rho=1, theta=np.pi / 180, threshold=70,
                                      minLineLength=25,
                                      maxLineGap=80)

    if lines_candidate is not None:
        slope = 0
        intercept_oY = 0
        for line in lines_candidate:
            y1cv, x1cv, y2cv, x2cv = line[0]
            cv2.line(frame, (y1cv, x1cv), (y2cv, x2cv), (0, 0, 255), 2)
            coeff = np.polynomial.polynomial.polyfit((y1cv, y2cv), (x1cv, x2cv), 1)
            if coeff is not None:
                slope += coeff[1]
                intercept_oY += coeff[0]
                # print("y_cv = {}*x_cv + {}".format(coeff[1], coeff[0]))
        slope /= len(lines_candidate)
        intercept_oY /= len(lines_candidate)
        print("y_cv = {}*x_cv + {}".format(slope, intercept_oY))

    cv2.imshow("Canny", frame_canny)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    _, frame = cap.read()
