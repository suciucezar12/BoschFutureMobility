import time

import cv2
import keyboard

cap = cv2.VideoCapture(0)


i = 0

while True:

    _, frame = cap.read()
    # if keyboard.is_pressed('a'):
    #     if was_pressed == False:
    #         i = i + 1
    #         print(i)
    #         cv2.imwrite("./sample_lane_detection/sample_" + str(i) + ".jpg", frame)
    #         print("Sample_" + str(i) + " added!")
    #         was_pressed = True
    #     else:
    #         was_pressed = False
    #
    # if keyboard.is_pressed('x'):
    #     exit()

    i = i + 1
    print(i)
    cv2.imwrite("./sample_lane_detection/sample_" + str(i) + ".jpg", frame)
    print("Sample_" + str(i) + " added!")
    cv2.imshow("Frame", frame)
    time.sleep(1)
    cv2.waitKey(1)