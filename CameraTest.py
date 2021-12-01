import cv2
import time

import keyboard

time.sleep(0.1)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    print(frame.size())
    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    if keyboard.is_pressed('q'):
        print("Pressed 'q' => stopped streaming.")
        break

cv2.destroyAllWindows()

