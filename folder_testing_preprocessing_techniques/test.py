import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame_ROI = frame[270:, :]
    cv2.imshow("ROI", frame_ROI)
    cv2.waitKey(1)
    