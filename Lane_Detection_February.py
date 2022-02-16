import copy
import time
import cv2
import numpy as np


class LaneDetection:

    def __init__(self):

        time.sleep(0.1)
        self.cap = cv2.VideoCapture(0)

        # trapezoid's coordinates
        self.y_top_left_trapezoid = 90
        self.x_top_trapezoid = 300
        self.y_top_right_trapezoid = 550

        # size of frame
        self.width_frame = 640
        self.height_frame = 480

    def get_warp(self, frame):  # from trapezoid to rectangle

        # cv2.imshow("Frame_Cropped", frame_cropped)

        source_coords = np.float32([(0, 480), (self.y_top_left_trapezoid, self.x_top_trapezoid),
                         (self.y_top_right_trapezoid, self.x_top_trapezoid), (640, 480)])

        destination_coords = np.float32([(0, 480), (0, self.x_top_trapezoid), (640, self.x_top_trapezoid), (640, 480)])

        perspective_correction = cv2.getPerspectiveTransform(source_coords, destination_coords)  # the transformation matrix

        print(perspective_correction)
        print("\n-----------------------------------------")
        print(source_coords)
        print("\n-----------------------------------------")
        print(destination_coords)

        warp_size = (self.width_frame, self.height_frame - self.x_top_trapezoid, )

        print(warp_size)
        print("\n-----------------------------------------")


        return cv2.warpPerspective(frame, perspective_correction, warp_size, flags=cv2.INTER_LANCZOS4)

    def drawROI(self, frame):   # draw ROI
        cv2.line(frame, (0, 480), (self.y_top_left_trapezoid, self.x_top_trapezoid), (0, 255, 0), 2)
        cv2.line(frame, (self.y_top_left_trapezoid, self.x_top_trapezoid), (self.y_top_right_trapezoid, self.x_top_trapezoid), (0, 255, 0), 2)
        cv2.line(frame, (self.y_top_right_trapezoid, self.x_top_trapezoid), (640, 480), (0, 255, 0), 2)

    def preprocessing(self, frame):

        frame_copy = copy.deepcopy(frame)
        frame_copy = frame_copy[self.x_top_trapezoid:, :]   # section from horizontal line = top of trapezoid

        # gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        #
        # blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
        #
        # canny_frame = cv2.Canny(blurred_frame, 100, 200)

        return frame_copy

    def run(self):

        ret, frame = self.cap.read()

        while True:

            # Selecting ROI -> looking for a trapezoid where our lanes would always appear
            # base of the trapezoid is actually the bottom line of our frame
            # self.drawROI(frame) # draw ROI

            processed_frame = self.preprocessing(frame)

            # cv2.imshow("Canny", processed_frame)

            warp_frame = self.get_warp(frame)

            cv2.imshow("Warp", warp_frame)

            self.drawROI(frame)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

            ret, frame = self.cap.read()


LD = LaneDetection()

LD.run()
