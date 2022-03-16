import cv2
import numpy as np

class LDTest:

    def __init__(self):
        self.x_cv_ROI = 270
        self.cap = cv2.VideoCapture(0)

    def preprocess(self, frame_ROI):
        gray = cv2.cvtColor(frame_ROI, code=cv2.COLOR_BGR2GRAY)
        contrast = cv2.convertScaleAbs(gray, alpha=1.3, beta=0)

        # cv2.imshow("ROI preprocessed", gray)
        # cv2.imshow("Contrast", contrast)

        canny_gray = cv2.Canny(gray, 30, 255)
        canny_contrast = cv2.Canny(contrast, 30, 255)

        cv2.imshow("ROI preprocessed", canny_gray)
        cv2.imshow("Contrast", canny_contrast)

    def run(self):

        ret, frame = self.cap.read()
        while True:
            frame_ROI = frame[self.x_cv_ROI:, :]
            self.preprocess(frame_ROI)
            cv2.imshow("ROI", frame_ROI)
            cv2.waitKey(1)
            _, frame = self.cap.read()

testld = LDTest()
testld.run()