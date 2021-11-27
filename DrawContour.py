import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow("frame", frame)
cv2.waitKey(0)
h = frame.shape[0]
w = frame.shape[1]
print("w = " + w + " h = " + h)
