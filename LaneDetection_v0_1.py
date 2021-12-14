import time
import cv2
import numpy as np


class LaneDetection:

    def __init__(self):
        time.sleep(0.1)
        self.cap = cv2.VideoCapture(0)

    def apply_filter(self, ret, frame):
        start = time.time()
        frame_copy = frame.copy()
        cv2.imshow("Frame", frame_copy)

        frame_copy = frame[int(frame.shape[0] / 2):, :]
        # cv2.imshow("Frame", frame_copy)

        # rgb -> grayscale
        frame_grayscale = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Grayscale", frame_grayscale)

        # Apply Gaussian Blur
        frame_blurred = cv2.GaussianBlur(frame_grayscale, (7, 7), cv2.BORDER_DEFAULT)
        # cv2.imshow("Gaussian Blur", frame_blurred)

        # Apply Canny
        frame_edge_detection = cv2.Canny(frame_blurred, 100, 200)
        cv2.imshow("Canny", frame_edge_detection)

        end = time.time()
        # print(end - start)

        # cv2.waitKey(1)
        return frame_edge_detection

    def run(self):
        ret, frame = self.cap.read()
        while ret:
            frame_edge_detection = self.apply_filter(ret, frame)
            # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap) second argument = P
            # third argument = Theta accuracies ?
            # fourth argument =  Threshold = Minimum length of line that should be detected = Nb of pixels which belong to that line
            # fifth argument = max allowed gap between line segments to treat them as single line

            lines = cv2.HoughLinesP(lines, frame_edge_detection, rho=1, theta=np.pi / 180, threshold=100, minLineLength=10, maxLineGap=100)
            frame_copy = frame_copy = frame[int(frame.shape[0] / 2):, :]
            tolerance = 25
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame_copy, (x1 - tolerance, y1), (x1 + tolerance, y1), (0, 255, 0), 3)
                    cv2.line(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
                cv2.imshow("Probabilistic Hough Transform", frame_copy)
            cv2.waitKey(1)

            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
