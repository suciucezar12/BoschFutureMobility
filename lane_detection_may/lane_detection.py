import copy
from threading import Thread
import time
import math
import numpy as np
import cv2
from utils import *


class LaneDetection():

    def __init__(self):
        """

        :param inP_img: receives a preprocessed image from a pipe
        :param outP_lane: outputs the result of the detection through the pipe
        """
        self.cap = cv2.VideoCapture(0)
        self.list_of_frames = []

        self.left_lines = []
        self.right_lines = []
        self.horizontal_lines = []
        self.left_lane = None
        self.right_lane = None
        self.road_lane = None

        ''' Info about previous detected road lanes'''
        self.previous_left_lane = []
        self.previous_right_lane = []

        self.utils = Utils()

        ''' Info about frame'''
        self.width = 640
        self.height = 480

        ''' Info about ROI '''
        self.x_cv_ROI = 270
        self.height_ROI = self.height - self.x_cv_ROI
        self.width_ROI = self.width

        ''' Info for IPM (Inverse Perspective Mapping)'''
        self.src_points_DLT = np.array(
            [[0, 0], [self.width_ROI, 0], [self.width_ROI, self.height_ROI], [0, self.height_ROI]],
            dtype=np.float32)
        self.dst_points_DLT = np.array([[0, 3.1], [66.7, 0], [53.5, 24.6], [11, 25.6]])  # expressed in centimeters
        self.pixel_resolution = 0.122  # centimeters per pixel
        self.H = self.utils.get_homography_matrix(self.src_points_DLT, self.dst_points_DLT,
                                                  self.pixel_resolution)
        self.inv_H = np.linalg.inv(self.H)
        self.height_ROI_IPM = 210  # calculated related to pixel_resolution and the real dimensions
        self.width_ROI_IPM = 547

        self.offset_origin = -20  # to correct the inclination of our camera
        self.y_heading_car_cv = self.width_ROI_IPM // 2 + self.offset_origin
        self.width_road_IPM = 310
        ''' ================================================================================================================================ '''

    def filter_lines(self, lines_candidate, frame_ROI, frame_ROI_IPM=None):
        left_lines = []
        right_lines = []
        horizontal_lines = []
        # used for intercept_oX criteria
        y_cv_margin = 80  # offset wrt to the center vertical line
        margin_y_cv_left = int(self.width_ROI / 2) - y_cv_margin
        margin_y_cv_right = int(self.width_ROI / 2) + y_cv_margin


        for line in lines_candidate:
            y1_cv, x1_cv, y2_cv, x2_cv = line[0]
            # centroid = [(y1_cv + y2_cv) // 2, (x1_cv + x2_cv) // 2]
            if abs(y1_cv - y2_cv) > 10:
                coeff = np.polynomial.polynomial.polyfit((y1_cv, y2_cv), (x1_cv, x2_cv), deg=1)
                # ---------------------------------
                # coeff = []
                # try:
                #     slope = (x2_cv - x1_cv) / (y2_cv - y1_cv)
                # except OverflowError:
                #     slope = 10000
                # coeff.append(y1_cv - slope * x1_cv)
                # coeff.append(slope)
                # print(coeff)
                # ---------------------------------
                if coeff is not None:
                    # coeff[1] -> slope in XoY coordinates
                    # coeff[0] -> intercept_oY in XoY coordinates
                    if coeff[1] != 10000:
                        if abs(coeff[1]) >= 0.7:  # slope = +-0.2 -> +-11.3 degrees
                            # OverFlowError when we get horizontal lines
                            try:
                                # intercept_oX = - int(coeff[0] / coeff[1])
                                # print((self.height_ROI - coeff[0]) / coeff[1])
                                intercept_oX = int((self.height_ROI - coeff[0]) / coeff[1])
                            except OverflowError:
                                intercept_oX = 30000  # some big value
                            # print("y = {}*x + {}".format(coeff[1], coeff[0]))
                            # print(intercept_oX)
                            if 0 <= intercept_oX <= margin_y_cv_left:  # left line
                                left_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                self.left_lines.append(line)
                                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 0, 0), 1)

                            if margin_y_cv_right <= intercept_oX <= self.width_ROI:  # right line
                                right_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                self.right_lines.append(line)
                                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 0, 255), 1)

                            # check by theta and intercept_oX (last criteria)
                            if coeff[1] <= -0.2:  # candidate left line
                                if 0 <= intercept_oX <= margin_y_cv_right:
                                    left_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                    self.left_lines.append(line)
                                    cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 0, 0), 1)

                            if coeff[1] >= 0.2:  # candidate right line
                                if margin_y_cv_left <= intercept_oX <= self.width_ROI:
                                    right_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                    self.right_lines.append(line)
                                    cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 0, 255), 1)
                        else:
                            if abs(coeff[1]) <= 0.3:
                                horizontal_lines.append(line)
                                self.horizontal_lines.append(line)

        return left_lines, right_lines, horizontal_lines

    def first_detection(self, frame_ROI, frame_ROI_IPM=None):
        # preprocessing step
        frame_ROI_preprocessed = self.utils.preprocessing(frame_ROI)
        # apply Probabilistic Hough Line
        lines_candidate = cv2.HoughLinesP(frame_ROI_preprocessed, rho=1, theta=np.pi / 180, threshold=40,
                                          minLineLength=25,
                                          maxLineGap=80)
        # filter lines
        # left_lines and right_lines are type of class Line
        # horizontal_lines are just lists of coordinates
        if lines_candidate is not None:
            left_lines, right_lines, horizontal_lines = self.filter_lines(lines_candidate, frame_ROI, frame_ROI_IPM)

            left_lane = self.utils.estimate_lane(left_lines, self.height_ROI, frame_ROI, frame_ROI_IPM)
            right_lane = self.utils.estimate_lane(right_lines, self.height_ROI, frame_ROI, frame_ROI_IPM)

            return left_lane, right_lane, horizontal_lines

        return None, None, None

    def optimized_detection(self, frame_ROI, framre_ROI_IPM=None):
        return None, None

    def optimized_intersection_detection(self, frame_ROI, left_lane, right_lane, frame_ROI_IPM=None):

        return None

    def get_offset_theta(self, frame_ROI, left_lane=None, right_lane=None, frame_ROI_IPM=None):
        offset = None
        theta = None
        intersection = False
        # we get all coordinates in IPM we need of our lanes
        left_lane_IPM = None
        right_lane_IPM = None
        if left_lane and right_lane:  # we have both lanes
            left_lane_IPM = self.utils.get_line_IPM(left_lane, self.H)
            right_lane_IPM = self.utils.get_line_IPM(right_lane, self.H)
            pass
        else:
            if left_lane is not None:  # only have our left lane
                left_lane_IPM = self.utils.get_line_IPM(left_lane, self.H)
                right_lane_IPM = self.utils.translation_IPM(left_lane_IPM, self.width_road_IPM, True)
                right_lane = self.utils.get_line_IPM(right_lane_IPM, self.inv_H)
                y1_cv, x1_cv, y2_cv, x2_cv = right_lane
                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)
            else:
                if right_lane is not None:  # only have our right lane
                    right_lane_IPM = self.utils.get_line_IPM(right_lane, self.H)
                    left_lane_IPM = self.utils.translation_IPM(right_lane_IPM, self.width_road_IPM, False)
                    left_lane = self.utils.get_line_IPM(left_lane_IPM, self.inv_H)
                    y1_cv, x1_cv, y2_cv, x2_cv = left_lane
                    cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)

        if left_lane_IPM is not None and right_lane_IPM is not None:
            y1_left_cv, x1_left_cv, y2_left_cv, x2_left_cv = left_lane_IPM
            y1_right_cv, x1_right_cv, y2_right_cv, x2_right_cv = right_lane_IPM
            if frame_ROI_IPM is not None:
                cv2.line(frame_ROI_IPM, (y1_left_cv, x1_left_cv), (y2_left_cv, x2_left_cv), (0, 255, 0), 3)
                cv2.line(frame_ROI_IPM, (y1_right_cv, x1_right_cv), (y2_right_cv, x2_right_cv), (0, 255, 0), 3)

            # theta
            y_heading_road_cv = (y2_left_cv + y2_right_cv) // 2
            x_heading_road_cv = (x2_left_cv + x2_right_cv) // 2

            y_bottom_road_cv = (y1_left_cv + y1_right_cv) // 2
            x_bottom_road_cv = (x1_left_cv + x1_right_cv) // 2
            cv2.line(frame_ROI_IPM, (y_heading_road_cv, x_heading_road_cv), (y_bottom_road_cv, x_bottom_road_cv),
                     (255, 255, 255), 3)
            road_line_reference_IPM = self.utils.get_line_IPM(
                [y_heading_road_cv, x_heading_road_cv, self.y_heading_car_cv, self.height_ROI_IPM], self.inv_H)
            y1_cv, x1_cv, y2_cv, x2_cv = road_line_reference_IPM
            self.road_lane = road_line_reference_IPM
            cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 255, 255), 3)

            # print("{}, {}, {}, {}".format(y_heading_road_cv, self.y_heading_car_cv, x_heading_road_cv, self.height_ROI_IPM))

            # centroid_road = [(y_heading_road_cv + y_bottom_road_cv) / 2, (x_heading_road_cv + x_bottom_road_cv) / 2]

            theta = math.degrees(
                math.atan((y_heading_road_cv - self.y_heading_car_cv) / (x_heading_road_cv - self.height_ROI_IPM)))
            # theta = math.degrees(
            #     math.atan((centroid_road[0] - self.y_heading_car_cv) / (centroid_road[1] - self.height_ROI_IPM)))

            # offset
            # print("{}, {}".format(y_bottom_road_cv, y_heading_road_cv))
            offset = (y_bottom_road_cv - self.y_heading_car_cv) * self.pixel_resolution

        return offset, theta, left_lane_IPM, right_lane_IPM

    def intersection_detection(self, frame_ROI, horizontal_lines, left_line_IPM, right_line_IPM, frame_ROI_IPM=None):
        # bounding box for filtering horizontal lines
        y1_left_cv, x1_left_cv, y2_left_cv, x2_left_cv = left_line_IPM
        y1_right_cv, x1_right_cv, y2_right_cv, x2_right_cv = right_line_IPM
        coeff_left_line_IPM = np.polynomial.polynomial.polyfit((y1_left_cv, y2_left_cv), (x1_left_cv, x2_left_cv), deg=1)
        coeff_right_line_IPM = np.polynomial.polynomial.polyfit((y1_right_cv, y2_right_cv), (x1_right_cv, x2_right_cv),
                                                               deg=1)
        margin_error = 25
        y_left_box = y1_left_cv - margin_error
        y_right_box = y1_right_cv + margin_error
        sum = 0
        x_bottom_ROI = 140
        x_points = []
        y_points = []
        slope_horiz = 0
        # cv2.line(frame_ROI_IPM, (y_left_box, 0), (y_left_box, self.height_ROI_IPM), (255, 0, 0), 5)
        # cv2.line(frame_ROI_IPM, (y_right_box, 0), (y_right_box, self.height_ROI_IPM), (255, 0, 0), 5)
        for line in horizontal_lines:
            y1_cv, x1_cv, y2_cv, x2_cv = line[0]
            line_IPM = self.utils.get_line_IPM(line[0], self.H)
            y1_IPM_cv, x1_IPM_cv, y2_IPM_cv, x2_IPM_cv = line_IPM
            if y_left_box <= y1_IPM_cv and y2_IPM_cv <= y_right_box:
                if x1_IPM_cv <= x_bottom_ROI and x2_IPM_cv <= x_bottom_ROI:
                    cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 255, 0), 2)
                    if frame_ROI_IPM is not None:
                        cv2.line(frame_ROI_IPM, (y1_IPM_cv, x1_IPM_cv), (y2_IPM_cv, x2_IPM_cv), (255, 255, 0), 2)
                    sum += math.sqrt((y2_IPM_cv - y1_IPM_cv) ** 2 + (x2_IPM_cv - x1_IPM_cv) ** 2)
                    coeff = np.polynomial.polynomial.polyfit((y1_IPM_cv, y2_IPM_cv), (x1_IPM_cv, x2_IPM_cv), deg=1)
                    # intersection with left and right lanes
                    print(coeff_left_line_IPM)
                    print(coeff_right_line_IPM)
                    print(coeff)
                    # left lane
                    y_cv, x_cv = self.utils.line_intersection(coeff, coeff_left_line_IPM)
                    cv2.circle(frame_ROI_IPM, (y_cv, x_cv), 2, (255, 255, 255), 2)
                    print("y_cv, x_cv = {}, {}".format(y_cv, x_cv))
                    x_points.append(y_cv)
                    y_points.append(x_cv)
                    # right lane
                    y_cv, x_cv = self.utils.line_intersection(coeff, coeff_right_line_IPM)
                    cv2.circle(frame_ROI_IPM, (y_cv, x_cv), 2, (255, 255, 255), 2)
                    print("y_cv, x_cv = {}, {}".format(y_cv, x_cv))
                    x_points.append(y_cv)
                    y_points.append(x_cv)
                    # x_points.append(y1_IPM_cv)
                    # x_points.append(y2_IPM_cv)
                    # x_points.append((y1_IPM_cv + y2_IPM_cv) / 2)
                    # y_points.append(x1_IPM_cv)
                    # y_points.append(x2_IPM_cv)
                    # y_points.append((x1_IPM_cv + x2_IPM_cv) / 2)
                    slope_horiz += coeff[1]
            # else:
                # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 255), 2)
                # if frame_ROI_IPM is not None:
                #     cv2.line(frame_ROI_IPM, (y1_IPM_cv, x1_IPM_cv), (y2_IPM_cv, x2_IPM_cv), (0, 255, 255), 2)
                # self.utils.draw_line(line, (0, 255, 255), frame_ROI)
                # if frame_ROI_IPM is not None:
                    # cv2.line(frame_ROI_IPM, (y1_IPM_cv, x1_IPM_cv), (y2_IPM_cv, x2_IPM_cv), (255, 255, 0), 2)
                    # self.utils.draw_line([line_IPM], (0, 255, 255), frame_ROI_IPM)
        # print(sum)
        if sum > 200:
            slope_horiz /= len(horizontal_lines)
            coeff = np.polynomial.polynomial.polyfit(x_points, y_points, deg=1)
            # y = coeff[1] * x + coeff[0]
            x1_cv = int(coeff[1] * 0 + coeff[0])
            x2_cv = int(coeff[1] * self.width_ROI_IPM + coeff[0])
            # cv2.line(frame_ROI_IPM, (0, abs(x1_cv - self.height_ROI_IPM)), (self.width_ROI_IPM, abs(x2_cv - self.height_ROI_IPM)), (0, 255, 0), 3)
            cv2.line(frame_ROI_IPM, (0, x1_cv),
                     (self.width_ROI_IPM, x2_cv), (0, 255, 0), 3)
            # print("slope horiz line = {}".format(coeff[1]))
            # theta_horizontal_lane = math.degrees(math.atan(coeff[1]))
            # theta_yaw_map = 90 + theta_horizontal_lane
            # print("theta_horiz = {}".format(theta_horizontal_lane))
            theta_yaw_map = 90 + math.degrees(slope_horiz)
            print("theta_yaw_map = {}".format(theta_yaw_map))

            return True, True
        else:
            return False, False

    def lane_detection(self, frame_ROI, frame_ROI_IPM=None):
        offset = None
        theta = None
        intersection = False

        if self.previous_left_lane and self.previous_right_lane:
            # TO DO: implement algorithm using data from previous frames for optimization
            left_lane, right_lane = self.optimized_detection(frame_ROI, frame_ROI_IPM)
            offset, theta, left_lane_IPM, right_lane_IPM = self.get_offset_theta(frame_ROI, left_lane, right_lane,
                                                                                 frame_ROI_IPM)
            intersection = self.optimized_intersection_detection(frame_ROI, left_lane_IPM, right_lane_IPM, frame_ROI_IPM)
        else:
            left_lane, right_lane, horizontal_lines = self.first_detection(frame_ROI, frame_ROI_IPM)
            offset, theta, left_lane_IPM, right_lane_IPM = self.get_offset_theta(frame_ROI, left_lane, right_lane,
                                                                                 frame_ROI_IPM)
            if left_lane_IPM is not None and right_lane_IPM is not None:
                self.left_lane = left_lane
                self.right_lane = right_lane
                if len(horizontal_lines):
                    intersection, horizontal_lane = self.intersection_detection(frame_ROI, horizontal_lines, left_lane_IPM, right_lane_IPM,
                                                               frame_ROI_IPM)
                    if horizontal_lane is not None:
                        if frame_ROI_IPM is not None:
                            cv2.line(frame_ROI_IPM, (self.y_heading_car_cv, 0), (self.y_heading_car_cv, self.height_ROI_IPM), (255, 255, 255), 3)

        return theta, offset, intersection

    def run(self):
        theta_prev = 0
        offset_prev = 0

        ret, frame = self.cap.read()

        while True:
            start = time.time()

            frame_ROI = frame[self.x_cv_ROI:, :]
            frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM),
                                                flags=cv2.INTER_NEAREST)
            # frame_ROI_IPM = None
            theta, offset, intersection = self.lane_detection(frame_ROI, frame_ROI_IPM=frame_ROI_IPM)

            if offset is not None:
                #print("OFFSET = {} cm".format(offset))
                offset_prev = offset
            # print("offset = {}".format(offset_prev))

            if theta is not None:
                #print("THETA = {}".format(theta))
                theta_prev = theta
            # print("theta = {}".format(theta_prev))

            # print("INTERSECTION = {}".format(intersection))

            cv2.imshow("ROI", frame_ROI)
            if frame_ROI_IPM is not None:
                cv2.imshow("IPM", frame_ROI_IPM)
            cv2.waitKey(1)
            # theta_prev = (theta_prev // 3) * 3


            end = time.time()
            print("time = {}".format(end - start))
            print("-----------------------------------------------------------")
            # if config.PRINT_EXEC_TIMES:
            #     print("Lane detection time: {}".format(end - start))
            #
            # ######### here the lane detection ends ###########
            #
            # lane_info = {"theta": -theta_prev, "offset": offset_prev, "horiz_line": intersection}
            #
            # self.outP_lane.send((end, lane_info, self.left_lane, self.right_lane, self.road_lane))   # sends the results of the detection back

            _, frame = self.cap.read()


LD = LaneDetection()
LD.run()
