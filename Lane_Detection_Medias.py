import time

import cv2
import numpy as np


class LaneDetection:

    def __init__(self):
        time.sleep(0.2)  # let camera warm-up
        self.cap = cv2.VideoCapture(0)

    def run(self):
        ret, frame = self.cap.read()
        while ret:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
