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
        y_cv_margin = 145  # offset wrt to the center vertical line
        margin_y_cv_left = int(self.width_ROI / 2) - y_cv_margin
        margin_y_cv_right = int(self.width_ROI / 2) + y_cv_margin

        for line in lines_candidate:
            y1_cv, x1_cv, y2_cv, x2_cv = line[0]
            coeff = np.polynomial.polynomial.polyfit((y1_cv, y2_cv), (x1_cv, x2_cv), deg=1)
            if coeff is not None:
                # coeff[1] -> slope in XoY coordinates
                # coeff[0] -> intercept_oY in XoY coordinates
                if abs(coeff[1]) >= 0.2:  # slope = +-0.2 -> +-11.3 degrees
                    # OverFlowError when we get horizontal lines
                    try:
                        intercept_oX = - (coeff[0] // coeff[1])
                    except OverflowError:
                        intercept_oX = 30000  # some big value

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
        left_lines, right_lines, horizontal_lines = self.filter_lines(lines_candidate, frame_ROI, frame_ROI_IPM)

        left_lane = self.utils.estimate_lane(left_lines, self.height_ROI, frame_ROI, frame_ROI_IPM)
        right_lane = self.utils.estimate_lane(right_lines, self.height_ROI, frame_ROI, frame_ROI_IPM)

        return left_lane, right_lane, horizontal_lines

    def optimized_detection(self, frame_ROI, framre_ROI_IPM=None):
        return None, None

    def optimized_intersection_detection(self, frame_ROI, left_lane, right_lane, frame_ROI_IPM=None):
        return None

    def get_offset_theta(self, frame_ROI, left_lane=None, right_lane=None, frame_ROI_IPM=None):
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

        theta = math.degrees(
            math.atan((y_heading_road_cv - self.y_heading_car_cv) / (x_heading_road_cv - self.height_ROI_IPM)))

        # offset
        offset = (y_bottom_road_cv - self.y_heading_car_cv) * self.pixel_resolution

        return offset, theta, left_lane_IPM, right_lane_IPM

    def intersection_detection(self, frame_ROI, horizontal_lines, left_lane, right_lane, frame_ROI_IPM):
        return None

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
            intersection = self.intersection_detection(frame_ROI, horizontal_lines, left_lane_IPM, right_lane_IPM,
                                                       frame_ROI_IPM)

        return theta, offset, intersection

    def run(self):
        ret, frame = self.cap.read()
        while True:
            start = time.time()

            # select our ROI
            frame_ROI = frame[self.x_cv_ROI:, :]
            frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM),
                                                flags=cv2.INTER_NEAREST)
            # frame_ROI_IPM = None
            theta, offset, intersection = self.lane_detection(frame_ROI, frame_ROI_IPM=frame_ROI_IPM)

            if offset is not None:
                print("OFFSET = {} cm".format(offset))
            if theta is not None:
                print("THETA = {}".format(theta))
            print("INTERSECTION = {}".format(intersection))

            # cv2.imshow("Frame", frame)
            cv2.imshow("IPM", frame_ROI_IPM)
            cv2.imshow("ROI", frame_ROI)
            cv2.waitKey(1)

            end = time.time()
            print("TIME = {}".format(end - start))
            print("---------------------------------------------------------")

            _, frame = self.cap.read()


LD = LaneDetection()
LD.run()
