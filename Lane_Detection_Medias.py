import time

import cv2
import numpy as np


class SlideWindow:
    # dimensions
    margin = None
    height = None

    # center point of down border
    x_center = None
    # y_center = None


class LaneDetection:

    def __init__(self):
        time.sleep(0.2)  # let camera warm-up
        self.cap = cv2.VideoCapture(0)

    def draw_box(self, x1, y1, x2, y2, x3, y3, x4, y4, image):
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(image, (x2, y2), (x3, y3), (0, 0, 255), 3)
        cv2.line(image, (x3, y3), (x4, y4), (0, 0, 255), 3)
        cv2.line(image, (x4, y4), (x1, y1), (0, 0, 255), 3)

    # method used to obtain the starting points (x) of left and right lane
    def get_starting_points_lanes(self, blurred_frame):

        partial_frame = blurred_frame[blurred_frame.shape[0] // 5:, :]

        # apply other filters to obtain better frame for histogram calculation

        histogram = np.sum(partial_frame, axis=0)  # sum on each column (on each particular column) => histogram

        size = len(histogram)
        x_left = np.argmax(histogram[0: size // 2])  # choose x_left
        x_right = np.argmax(histogram[size // 2:]) + size // 2  # choose x_right

        # print(f"histogram[{x_left}] = " + histogram[x_left])
        if histogram[x_left] == 0:  # no left lane
            x_left = None

        # print(f"histogram[{x_right}] = " + histogram[x_right])
        if histogram[x_right] == 0:  # no right lane
            x_right = None

        return x_left, x_right

        # apply filters

    def preprocessing_frame(self, frame):

        frame_copy = frame.copy()

        height = frame_copy.shape[0]
        frame_copy = frame_copy[int(0.65 * height):, :]  # crop image (select our ROI)

        # Apply filters
        gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), cv2.BORDER_DEFAULT)
        # edge_frame = cv2.Canny(blurred_frame, 100, 200)
        return blurred_frame

    # method for obtaining lanes
    def get_lanes(self, edge_frame):
        x_left, x_right = self.get_starting_points_lanes(edge_frame)

    def run(self):
        ret, frame = self.cap.read()

        while ret:
            # preprocessing frame
            blurred_frame = self.preprocessing_frame(frame)

            edge_frame = cv2.Canny(blurred_frame, 100, 200)

            cv2.imshow("Edge Detection", edge_frame)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
