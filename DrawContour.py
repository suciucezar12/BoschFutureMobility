import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.line(frame, (160, 240), (320, 240), (0, 255, 0)) # up base
cv2.line(frame, (320, 240), (480, 640), (0, 255, 0)) # right
cv2.line(frame, (480, 640), (480, 0), (0, 255, 0)) # low base
cv2.line(frame, (480, 0), (240, 160), (0, 255,0)) # left
cv2.imwrite("ImageFromCamera.jpg", frame)
h = frame.shape[0]
w = frame.shape[1]
print("w = " + str(w) + " h = " + str(h))
