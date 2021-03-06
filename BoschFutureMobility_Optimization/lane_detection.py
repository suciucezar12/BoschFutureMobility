import math

import cv2
import numpy as np
import time
from utils import *


class LaneDetection:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.utils = Utils()
        ''' Info about previous detected road lanes'''
        self.previous_left_lane = []
        self.previous_right_lane = []

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
                        if abs(coeff[1]) >= 0.2:  # slope = +-0.2 -> +-11.3 degrees
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
                                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 0, 0), 1)

                            if margin_y_cv_right <= intercept_oX <= self.width_ROI:  # right line
                                right_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 0, 255), 1)

                            # check by theta and intercept_oX (last criteria)
                            if coeff[1] <= -0.2:  # candidate left line
                                if 0 <= intercept_oX <= margin_y_cv_right:
                                    left_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                    cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 0, 0), 1)

                            if coeff[1] >= 0.2:  # candidate right line
                                if margin_y_cv_left <= intercept_oX <= self.width_ROI:
                                    right_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                    cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 0, 255), 1)
                        else:
                            horizontal_lines.append(line)

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

    def optimized_preprocessing(self, window_ROI, frame_ROI, frame_ROI_IPM=None):
        window_ROI_gray = cv2.cvtColor(window_ROI, cv2.COLOR_BGR2RGB)
        window_ROI_canny = cv2.Canny(window_ROI_gray, 180, 255)
        return window_ROI_canny
        pass

    def optimized_sliding_window_pipeline(self, coords, coeff, color, frame_ROI, frame_ROI_IPM=None):
        y_cv_min, x_cv_min, y_cv_max, x_cv_max = coords
        window_ROI = frame_ROI[x_cv_min:x_cv_max, y_cv_min:y_cv_max]
        window_ROI_preprocessed = self.optimized_preprocessing(window_ROI, frame_ROI, frame_ROI_IPM)
        theta = math.degrees(math.atan(coeff[1]))
        theta_margin = 15

        lines_candidate = lines_candidate = cv2.HoughLinesP(window_ROI_preprocessed, rho=1, theta=np.pi / 180, threshold=20,
                                          minLineLength=25,
                                          maxLineGap=80)

        # filter lines
        correct_lines = []
        if lines_candidate is not None:
            for line in lines_candidate:
                y1_cv, x1_cv, y2_cv, x2_cv = line[0]
                # translation from the origin of sliding window to the origin of frame_ROI
                line = [y1_cv + y_cv_min, x1_cv + x_cv_min, y2_cv + y_cv_min, x2_cv + x_cv_min]
                y1_cv, x1_cv, y2_cv, x2_cv = line
                if abs(y1_cv - y2_cv) >= 10:
                    coeff_line = np.polynomial.polynomial.polyfit((y1_cv, y2_cv), (x1_cv, x2_cv), 1)
                    if abs(theta - math.degrees(math.atan(coeff_line[1]))) <= theta_margin:
                        cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), color, 2)
                        correct_lines.append(Line(coords=line, coeff=coeff_line))




        # cv2.imshow("window_ROI", window_ROI_preprocessed)
        # cv2.waitKey(1000)
        return correct_lines
        pass

    def lane_optimized_detection(self, nb_of_windows, margin_left, margin_right, lane, frame_ROI, left_lane=False, frame_ROI_IPM=None):
        # generate sliding windows coordinates
        y1_cv, x1_cv, y2_cv, x2_cv = lane
        coeff = np.polynomial.polynomial.polyfit((y1_cv, y2_cv), (x1_cv, x2_cv), 1)
        y_cv_array = np.linspace(y1_cv, y2_cv, nb_of_windows+1)

        y_cv_points = []
        x_cv_points = []

        edge_lines = []

        for i in range(len(y_cv_array) - 1):
            y_cv = y_cv_array[i]
            yp_cv = y_cv_array[i+1]
            x_cv = coeff[1] * y_cv + coeff[0]
            xp_cv = coeff[1] * yp_cv + coeff[0]

            # extend by margin error
            y_cv_points.append(y_cv - margin_left)
            y_cv_points.append(y_cv + margin_right)
            y_cv_points.append(yp_cv - margin_left)
            y_cv_points.append(yp_cv + margin_right)

            x_cv_points.append(x_cv)
            x_cv_points.append(xp_cv)

            # clipping
            y_cv_points = [y_cv if y_cv >= 0 else 0 for y_cv in y_cv_points]
            y_cv_points = [y_cv if y_cv <= self.width_ROI else self.width_ROI for y_cv in y_cv_points]

            x_cv_points = [x_cv if x_cv >= 0 else 0 for x_cv in x_cv_points]
            x_cv_points = [x_cv if x_cv <= self.height_ROI else self.height_ROI for x_cv in x_cv_points]

            # get our corners
            x_cv_min, x_cv_max = int(min(x_cv_points)), int(max(x_cv_points))
            y_cv_min, y_cv_max = int(min(y_cv_points)), int(max(y_cv_points))

            # check if our sliding window is in within our image (previous detected lane was outside of our FOV)
            if x_cv_min != x_cv_max and y_cv_min != y_cv_max:
                # pipeline for detecting edge points -> return set of points
                if left_lane is False:
                    correct_lines = self.optimized_sliding_window_pipeline([y_cv_min, x_cv_min, y_cv_max, x_cv_max], coeff, (0, 0, 255), frame_ROI, frame_ROI_IPM)
                else:
                    correct_lines = self.optimized_sliding_window_pipeline([y_cv_min, x_cv_min, y_cv_max, x_cv_max],
                                                                           coeff, (255, 0, 0), frame_ROI, frame_ROI_IPM)
                cv2.rectangle(frame_ROI, (y_cv_min, x_cv_min), (y_cv_max, x_cv_max), (255, 0, 255), 2)
                for correct_line in correct_lines:
                    edge_lines.append(correct_line)
                # edge_lines.append(correct_lines)

            y_cv_points = []
            x_cv_points = []

        if len(edge_lines):
            # print(edge_lines)
            return self.utils.estimate_lane(edge_lines, self.height_ROI, frame_ROI, frame_ROI_IPM)
        else:
            return None

    def optimized_detection(self, frame_ROI, frame_ROI_IPM=None):
        nb_of_windows = 3
        left_lane = self.lane_optimized_detection(nb_of_windows, 50, 50, self.previous_left_lane, frame_ROI, left_lane=True, frame_ROI_IPM=frame_ROI_IPM)
        right_lane = self.lane_optimized_detection(nb_of_windows, 50, 50, self.previous_right_lane, frame_ROI, left_lane=False,  frame_ROI_IPM=frame_ROI_IPM)


        return left_lane, right_lane

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
            road_line_reference = self.utils.get_line_IPM(
                [y_heading_road_cv, x_heading_road_cv, self.y_heading_car_cv, self.height_ROI_IPM], self.inv_H)
            y1_cv, x1_cv, y2_cv, x2_cv = road_line_reference
            cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 255, 255), 3)

            # print("{}, {}, {}, {}".format(y_heading_road_cv, self.y_heading_car_cv, x_heading_road_cv, self.height_ROI_IPM))
            theta = math.degrees(
                math.atan((y_heading_road_cv - self.y_heading_car_cv) / (x_heading_road_cv - self.height_ROI_IPM)))

            # offset
            # print("{}, {}".format(y_bottom_road_cv, y_heading_road_cv))
            offset = (y_bottom_road_cv - self.y_heading_car_cv) * self.pixel_resolution

        return offset, theta, left_lane_IPM, right_lane_IPM

    def intersection_detection(self, frame_ROI, horizontal_lines, left_line_IPM, right_line_IPM, frame_ROI_IPM=None):
        # bounding box for filtering horizontal lines
        # print(left_line_IPM[0])
        # print(right_line_IPM[0])
        y1_left_cv, x1_left_cv, y2_left_cv, x2_left_cv = left_line_IPM
        y1_right_cv, x1_right_cv, y2_right_cv, x2_right_cv = right_line_IPM
        margin_error = 25
        y_left_box = y1_left_cv - margin_error
        y_right_box = y1_right_cv + margin_error
        sum = 0
        # cv2.line(frame_ROI_IPM, (y_left_box, 0), (y_left_box, self.height_ROI_IPM), (255, 0, 0), 5)
        # cv2.line(frame_ROI_IPM, (y_right_box, 0), (y_right_box, self.height_ROI_IPM), (255, 0, 0), 5)
        for line in horizontal_lines:
            y1_cv, x1_cv, y2_cv, x2_cv = line[0]
            line_IPM = self.utils.get_line_IPM(line[0], self.H)
            y1_IPM_cv, x1_IPM_cv, y2_IPM_cv, x2_IPM_cv = line_IPM
            if y_left_box <= y1_IPM_cv and y2_IPM_cv <= y_right_box:
                # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 255, 0), 2)
                self.utils.draw_line(line, (255, 255, 0), frame_ROI)
                if frame_ROI_IPM is not None:
                    # cv2.line(frame_ROI_IPM, (y1_IPM_cv, x1_IPM_cv), (y2_IPM_cv, x2_IPM_cv), (255, 255, 0), 2)
                    self.utils.draw_line([line_IPM], (255, 255, 0), frame_ROI_IPM)
                sum += math.sqrt((y2_IPM_cv - y1_IPM_cv) ** 2 + (x2_IPM_cv - x1_IPM_cv) ** 2)
            else:
                # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 255), 2)
                # if frame_ROI_IPM is not None:
                #     cv2.line(frame_ROI_IPM, (y1_IPM_cv, x1_IPM_cv), (y2_IPM_cv, x2_IPM_cv), (0, 255, 255), 2)
                self.utils.draw_line(line, (0, 255, 255), frame_ROI)
                if frame_ROI_IPM is not None:
                    # cv2.line(frame_ROI_IPM, (y1_IPM_cv, x1_IPM_cv), (y2_IPM_cv, x2_IPM_cv), (255, 255, 0), 2)
                    self.utils.draw_line([line_IPM], (0, 255, 255), frame_ROI_IPM)
        # print(sum)
        if sum > 300:
            return True
        else:
            return False

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
            self.previous_left_lane = left_lane
            self.previous_right_lane = right_lane
        else:
            left_lane, right_lane, horizontal_lines = self.first_detection(frame_ROI, frame_ROI_IPM)
            offset, theta, left_lane_IPM, right_lane_IPM = self.get_offset_theta(frame_ROI, left_lane, right_lane,
                                                                                 frame_ROI_IPM)
            if left_lane_IPM is not None and right_lane_IPM is not None:
                if len(horizontal_lines):
                    intersection = self.intersection_detection(frame_ROI, horizontal_lines, left_lane_IPM, right_lane_IPM,
                                                               frame_ROI_IPM)
                self.previous_left_lane = left_lane
                self.previous_right_lane = right_lane

        return theta, offset, intersection

    def run(self):
        ret, frame = self.cap.read()

        theta_prev = None
        offset_prev = None

        while True:
            start = time.time()

            # select our ROI
            frame_ROI = frame[self.x_cv_ROI:, :]
            # frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM),
            #                                     flags=cv2.INTER_NEAREST)
            frame_ROI_IPM = None
            theta, offset, intersection = self.lane_detection(frame_ROI, frame_ROI_IPM=frame_ROI_IPM)

            if offset is not None:
                print("OFFSET = {} cm".format(offset))
                offset_prev = offset

            if theta is not None:
                print("THETA = {}".format(theta))
                theta_prev = theta

            print("INTERSECTION = {}".format(intersection))

            # cv2.imshow("Frame", frame)
            # cv2.imshow("IPM", frame_ROI_IPM)
            cv2.imshow("ROI", frame_ROI)
            cv2.waitKey(1)

            end = time.time()
            print("TIME = {}".format(end - start))
            print("---------------------------------------------------------")

            _, frame = self.cap.read()


LD = LaneDetection()
LD.run()
