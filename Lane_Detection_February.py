import time
import cv2


class LaneDetection:

    def __init__(self):
        time.sleep(0.1)
        self.cap = cv2.VideoCapture(0)

    def run(self):

        frame = self.cap.read()

        cv2.imshow("Frame", frame)
        cv2.waitKey(0)

LD = LaneDetection()

LD.run()
