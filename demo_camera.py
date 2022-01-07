import cv2

time.sleep(2)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while ret:
	
	cv2.imshow("", frame)
	cv2.waitkey(1)
	
	re, frame = cap.read()