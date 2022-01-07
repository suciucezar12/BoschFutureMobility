import cv2
import time

time.sleep(0.2)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while ret:
	
	cv2.imshow("", frame)
	cv2.waitKey(1)
	
	re, frame = cap.read()