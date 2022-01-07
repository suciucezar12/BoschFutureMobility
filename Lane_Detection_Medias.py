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
        sobel_frame = cv2.Sobel(blurred_frame, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
        cv2.imshow("Sobel", sobel_frame)
        cv2.waitKey(1)
        partial_frame = sobel_frame[sobel_frame.shape[0] // 5:, :]
        histogram = np.sum(partial_frame, axis=0)  # sum on each column (on each particular column) => histogram
        # print(histogram)
        size = len(histogram)
        x_left = np.argmax(histogram[0: size // 2])  # choose x_left
        x_right = np.argmax(histogram[size // 2:]) + size // 2  # choose x_right

        if histogram[x_left] == 0:  # no left lane
            x_left = None

        if histogram[x_right] == 0:  # no right lane
            x_right = None

        return x_left, x_right

    # apply filters
    def preprocessing_frame(self, frame):
        frame_copy = frame.copy()

        width = frame_copy.shape[1]
        height = frame_copy.shape[0]
        frame_copy = frame_copy[int(0.5 * height):, :]  # crop image

        gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), cv2.BORDER_DEFAULT)

        self.get_starting_points_lanes(blurred_frame)   # use here to no compute 2 times all preprocessing phases until edge detection phase

        edge_frame = cv2.Canny(blurred_frame, 100, 200)

        return edge_frame

    # method for obtaining lanes
    def get_lanes(self, edge_frame):
        x_left, x_right = self.get_starting_points_lanes(edge_frame)

    def run(self):
        ret, frame = self.cap.read()

        while ret:
            edge_frame = self.preprocessing_frame(frame)  # apply filters and selecting ROI

            self.get_lanes(edge_frame)

            cv2.imshow("Preprocessing", edge_frame)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
