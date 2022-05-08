import cv2

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while True:
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    _, frame = cap.read()