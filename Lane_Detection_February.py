import time
import cv2


class LaneDetection:

    def __init__(self):
        time.sleep(0.1)
        self.cap = cv2.VideoCapture(0)

    def run(self):

        ret, frame = self.cap.read()

        while True:

            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

            ret, frame = self.cap.read()


LD = LaneDetection()

LD.run()
