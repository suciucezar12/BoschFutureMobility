import time
import numpy as np
import cv2


class LaneDetection:

    def __init__(self):
        time.sleep(0.1)
        self.cap = cv2.VideoCapture(0)

    def PreProcessing(self, frame):
        frame_copy = frame[3 * int(frame.shape[0] / 5):, :]

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
            frame_copy = frame[3 * int(frame.shape[0] / 5):, :]

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    slope = float((x2 - x1) / (y2 - y1))
                    if slope < 0.0:
                        left_lanes.append([x1, y1, x2, y2])
                    else:
                        right_lanes.append([x1, y1, x2, y2])

                for lane in left_lanes:
                    x1, y1, x2, y2 = lane
                    cv2.line(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)

                for lane in right_lanes:
                    x1, y1, x2, y2 = lane
                    cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.imshow("Frame", frame)
            cv2.imshow("PHT", frame_copy)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.Run()
