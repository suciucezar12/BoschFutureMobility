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
        for line in lines_candidate:
            y1cv, x1cv, y2cv, x2cv = line[0]
            cv2.line(frame, (y1cv, x1cv), (y2cv, x2cv), (0, 0, 255), 2)

    cv2.imshow("Canny", frame_canny)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    _, frame = cap.read()
