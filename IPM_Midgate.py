import cv2

x_cv_ROI = 270
cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while True:

    cv2.line(frame, (0, x_cv_ROI + 5), (640, x_cv_ROI), (0, 0, 255), 1)

    frame_ROI = frame[x_cv_ROI:, :]

    cv2.imshow("ROI", frame_ROI)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    ret, frame = cap.read()