import cv2
import numpy


class LaneDetection:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        pass

    def run(self):

        ret, frame = self.cap.read()
        while True:

            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            _, frame = self.cap.read()

        pass