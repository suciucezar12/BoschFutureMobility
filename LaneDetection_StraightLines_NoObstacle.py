import time
import numpy as np
import cv2


class Lane:

    # Y = slope * X + c         --> line equation
    # slope = float((y2 - y1) / (x2 - x1))      --> slope equation
    # c = Y(i) - slope * X(i)
    slope = 0.0
    coordinates = []    # x1, y1, x2, y2
    c = 0

    def __init__(self, coordinates, slope):
        self.slope = slope
        self.coordinates = coordinates
        self.c = int((coordinates[1] - slope * coordinates[0] + coordinates[3] - slope * coordinates[2]) / 2)
        pass


class LaneDetection:

    def __init__(self):
        time.sleep(0.1)
        self.cap = cv2.VideoCapture(0)

    def PreProcessing(self, frame):
        frame_copy = frame[int(int(frame.shape[0] * 0.55)):, :]

        frame_grayscale = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

        frame_blurred = cv2.GaussianBlur(frame_grayscale, (7, 7), cv2.BORDER_DEFAULT)

        frame_edge = cv2.Canny(frame_blurred, 100, 200)

        return frame_edge

    def Run(self):
        ret, frame = self.cap.read()
        while ret:
            frame_edge = self.PreProcessing(frame)

            lines = cv2.HoughLinesP(frame_edge, rho=1, theta=np.pi / 180, threshold=100, minLineLength=10,
                                    maxLineGap=100)
            left_lanes = []
            right_lanes = []
            frame_copy = frame[int(int(frame.shape[0] * 0.55)):, :]

            if lines is not None:
                # classify lanes based on their slope
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    slope = float((y2 - y1) / (x2 - x1))
                    if slope < 0.0:
                        left_lanes.append(Lane([x1, y1, x2, y2], slope))
                    else:
                        right_lanes.append(Lane([x1, y1, x2, y2], slope))

                # display candidates for left lane
                for lane in left_lanes:
                    x1, y1, x2, y2 = lane.coordinates
                    cv2.line(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)

                # display candidates for right lane
                for lane in right_lanes:
                    x1, y1, x2, y2 = lane.coordinates
                    cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)

            margin = 25

            # choose some x (about half of ROI)
            y = frame_edge.shape[0] / 2
            # for lane in left_lanes:



            cv2.imshow("Frame", frame)
            cv2.imshow("PHT Candidates", frame_copy)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.Run()
