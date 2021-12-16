import time
import numpy as np
import cv2


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

    def Averagelanes(self, lanes):
        x1_f = 0
        y1_f = 0
        x2_f = 0
        y2_f = 0

        for lane in lanes:
            x1, y1, x2, y2 = lane

            x1_f += x1
            y1_f += y1

            x2_f += x2
            y2_f += y2

        x1_f = int(x1_f / len(lanes))
        y1_f = int(y1_f / len(lanes))
        x2_f = int(x2_f / len(lanes))
        y2_f = int(y2_f / len(lanes))
        return [x1_f, y1_f, x2_f, y2_f]

    def drawLane(self, image, lane, color):
        x1, y1, x2, y2 = lane
        cv2.line(image, (x1, y1), (x2, y2), color, 3)

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
                        left_lanes.append([x1, y1, x2, y2])
                    else:
                        right_lanes.append([x1, y1, x2, y2])

                # # display candidates for left lane
                # for lane in left_lanes:
                #     x1, y1, x2, y2 = lane
                #     cv2.line(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
                #
                # # display candidates for right lane
                # for lane in right_lanes:
                #     x1, y1, x2, y2 = lane
                #     cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)

                left_lane = self.Averagelanes(left_lanes)
                right_lane = self.Averagelanes(right_lanes)

                # x1, y1, x2, y2 = left_lane
                # cv2.line(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
                self.drawLane(frame_copy, left_lane, (255, 0, 0))
                # x1, y1, x2, y2 = right_lane
                # cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
                self.drawLane(frame_copy, right_lane, (0, 0, 255))
            cv2.imshow("Frame", frame)
            cv2.imshow("PHT Candidates", frame_copy)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.Run()
