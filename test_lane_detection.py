import cv2
import numpy as np

class LDTest:

    def __init__(self):
        self.x_cv_ROI = 270
        self.cap = cv2.VideoCapture(0)

    def preprocess(self, frame_ROI):
        gray = cv2.cvtColor(frame_ROI, code=cv2.COLOR_BGR2GRAY)
        cv2.imshow("ROI preprocessed", gray)

    def run(self):

        ret, frame = self.cap.read()
        while True:
            frame_ROI = frame[self.x_cv_ROI:, :]
            self.preprocess(frame_ROI)
            cv2.imshow("ROI", frame_ROI)
            _, frame = self.cap.read()