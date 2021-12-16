import math
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
        cv2.line(image, (x1, y1), (x2, y2), color, 5)

    # def General_Equation_Coeffcients(self, x1, y1, x2, y2):
    #     a = float(y1 - y2)
    #     b = float(x2 - x1)
    #     c = float((x1 - x2) * y1 + (y2 - y1) * x1)
    #     return a, b, c
    #
    # def Angle_VanishingPoint(self, left_lane, right_lane, width):
    #     x1l, y1l, x2l, y2l = left_lane
    #     x1r, y1r, x2r, y2r = right_lane
    #     # swap
    #     x1l, x2l = x2l, x1l
    #     x1r, x2r = x2l, x2r
    #     # create general equation (a, b, c coefficients)
    #     a_l, b_l, c_l = self.General_Equation_Coeffcients(x2l, y2l, x1l, y1l)
    #     a_r, b_r, c_r = self.General_Equation_Coeffcients(x2r, y2r, x1r, y1r)
    #     # get vanishing point
    #     x = float(((b_l * c_r - b_r * c_l) / (a_l * b_r - a_r * b_l)))
    #     y = float(((c_l * a_r - c_r * a_l) / (a_l * b_r - a_r * b_l)))
    #     # (x, y) is under the image
    #     # create the angle between (x, y) (= (y, x) in opencv) and (0, width / 2) and vertical axis
    #     theta = - math.atan((y / (width / 2) - x))
    #     return theta
    #     pass

    def General_Equation_Form(self, x1, y1, x2, y2):
        A = (y1 - y2)
        B = (x1 - x2)
        C = (x1 * y2 - x2 * y1)
        return A, B, -C

    def Intersection(self, L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return False

    def Angle(self, left_lane, right_lane, width, height):
        x1l, y1l, x2l, y2l = left_lane
        y1l = -(y1l - height)
        y2l = -(y2l - height)

        x1r, y1r, x2r, y2r = right_lane
        y1r = -(y1r - height)
        y2r = -(y2r - height)

        L1 = self.General_Equation_Form(x1l, y1l, x2l, y2l)  # L1 -> left lane
        L2 = self.General_Equation_Form(x1r, y1r, x2r, y2r)  # l2 -> right lane

        x0, y0 = self.Intersection(L1, L2)

        print("x0 = " + str(x0) + ", y0 = " + str(y0))

        theta = math.atan(float((x0 - width / 2) / y0))
        return theta


    def Run(self):
        ret, frame = self.cap.read()
        while ret:
            frame_edge = self.PreProcessing(frame)

            lines = cv2.HoughLinesP(frame_edge, rho=1, theta=np.pi / 180, threshold=100, minLineLength=10,
                                    maxLineGap=100)
            left_lanes = []
            right_lanes = []
            frame_copy = frame[int(int(frame.shape[0] * 0.55)):, :]  # used for displaying

            if lines is not None:
                # classify lanes based on their slope
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    slope = float((y2 - y1) / (x2 - x1))
                    if slope < 0.0:
                        left_lanes.append([x1, y1, x2, y2])
                    else:
                        right_lanes.append([x1, y1, x2, y2])

                if len(left_lanes) and len(right_lanes):    # identify both lanes

                    left_lane = self.Averagelanes(left_lanes)
                    self.drawLane(frame_copy, left_lane, (255, 0, 0))

                    right_lane = self.Averagelanes(right_lanes)
                    self.drawLane(frame_copy, right_lane, (0, 0, 255))

                    theta = self.Angle(left_lane, right_lane, frame_copy.shape[0], frame_copy.shape[1])
                    print("theta = " + str(theta))
            # cv2.line(frame, (frame.shape[0] / 2, int(frame.shape[1])), (int(frame.shape[0] / 2), 0),
            #          (255, 255, 255), 3)
            # cv2.line(frame_copy, (int(frame_copy.shape[0] / 2), int(frame_copy.shape[1] / 2)), (int(frame_copy.shape[0] / 2), 0), (255, 255, 255), 3)
            cv2.imshow("Frame", frame)
            cv2.imshow("PHT", frame_copy)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.Run()
