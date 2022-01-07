import time

import cv2
import numpy as np


class SlideWindow:
    # dimensions
    margin = None
    height = None

    # center point of down border
    x0 = None
    y0 = None


class LaneDetection:

    def __init__(self):
        time.sleep(0.2)  # let camera warm-up
        self.cap = cv2.VideoCapture(0)

    # method used to obtain the starting points (x, 0) of left and right lane
    def find_starting_points_lanes(self, frame):
        partial_frame = frame[frame.shape[0] // 5:, :]
        histogram = np.sum(partial_frame, axis=0)  # sum on each column (on each particular column) => histogram
        size = len(histogram)
        x_left = np.argmax(histogram[0: size // 2])  # choose x_left
        x_right = np.argmax(histogram[size // 2:]) + size // 2  # choose x_right
        return x_left, x_right

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
            edge_frame = self.preprocessing_frame(frame)  # apply filters and selecting ROI
            self.find_starting_points_lanes(frame)

            cv2.imshow("Preprocessing", edge_frame)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
