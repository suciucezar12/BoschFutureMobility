import time

import cv2
import numpy as np

class Slide_Window():
    pass


class Lane_Detection:

    def __init__(self):
        time.sleep(0.2)  # let camera warm-up
        self.cap = cv2.VideoCapture(0)

    def preprocessing_frame(self, frame):
        frame_copy = frame.copy()

        width = frame_copy.shape[1]
        height = frame_copy.shape[0]
        frame_copy = frame_copy[int(0.5 * height):, :]  # crop image

        # Apply filters
        gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), cv2.BORDER_DEFAULT)
        edge_frame = cv2.Canny(blurred_frame, 100, 200)

        return edge_frame

    def run(self):
        ret, frame = self.cap.read()

        while ret:

            edge_frame = self.preprocessing_frame(frame)    # apply filters and selecting ROI

            cv2.imshow("Preprocessing", edge_frame)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = Lane_Detection()
LD.run()
