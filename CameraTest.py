import cv2
import time

import keyboard

time.sleep(1)

cap = cv2.VideoCapture(0)

# while True:

ret, frame = cap.read()
cv2.imshow("frame", frame)
cv2.waitKey(0)
# if keyboard.is_pressed('q'):
#     print("Pressed 'q' => stopped streaming.")

cv2.destroyAllWindows()

