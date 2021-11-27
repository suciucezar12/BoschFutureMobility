import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cv2.line(frame, (190, 260), (460, 260), (255, 0, 0))    # up base
cv2.line(frame, (460, 260), (640, 480), (0, 255, 0))    # right
# cv2.line(frame, (640, 480), (0, 480), (0, 0, 255))  # low base
# cv2.line(frame, (0, 480), (240, 160), (255, 255, 255))  # left
cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.imwrite("ImageFromCamera.jpg", frame)
h = frame.shape[0]
w = frame.shape[1]
print("w = " + str(w) + " h = " + str(h))
