import math
import time

import cv2
import numpy as np
from utils import Utils


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
        self.height_ROI_IPM = 210  # calculated related to pixel_resolution and the real dimensions
        self.width_ROI_IPM = 547

        self.offset_origin = -20  # to correct the inclination of our camera
        self.y_heading_car_cv = self.width_ROI_IPM // 2 + self.offset_origin

    def preprocessing(self, frame_ROI):
        frame_ROI_gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
        frame_ROI_blurred = cv2.GaussianBlur(frame_ROI_gray, (5, 5), 0)
        frame_ROI_canny = cv2.Canny(frame_ROI_blurred, 180, 255)
        return frame_ROI_canny

    def filter_lines(self, lines_candidate, frame_ROI, frame_ROI_IPM=None):
        horizontal_lines = []
        left_lines = []
        right_lines = []
        for line in lines_candidate:
            intercept_oX, theta = self.utils.get_intercept_theta(line)
            # check if horizontal line
            if abs(theta) <= 25:
                horizontal_lines.append(line)
            else:
                # check if left or right line
                line_code = self.utils.left_or_right_candidate_line(intercept_oX, theta)
                # left line
                if line_code == 0:
                    left_lines.append(line)
                    self.utils.draw_line(line, (255, 0, 0), frame_ROI)
                # right line
                if line_code == 1:
                    right_lines.append(line)
                    self.utils.draw_line(line, (0, 0, 255), frame_ROI)

        return left_lines, right_lines, horizontal_lines

    def first_detection(self, frame_ROI_preprocessed, frame_ROI, frame_ROI_IPM=None):
        """
        We use this algorithm when we don't have any previous data about the road
        :param frame_ROI:
        :param frame_ROI_IPM:
        :return:
        """
        lines_candidate = cv2.HoughLinesP(frame_ROI_preprocessed, rho=1, theta=np.pi / 180, threshold=45,
                                          minLineLength=25,
                                          maxLineGap=80)
        left_lane = None
        right_lane = None
        if lines_candidate is not None:
            left_lines, right_lines, horizontal_lines = self.filter_lines(lines_candidate, frame_ROI, frame_ROI_IPM)

            if left_lines:
                left_lane = self.utils.polyfit(left_lines, frame_ROI, left_lane=True)
            if right_lines:
                right_lane = self.utils.polyfit(right_lines, frame_ROI, left_lane=False)
        return left_lane, right_lane

    def get_offset_theta(self, frame_ROI, left_lane=None, right_lane=None, frame_ROI_IPM=None):
        # we get all coordinates in IPM we need of our lanes
        left_lane_IPM = None
        right_lane_IPM = None
        if left_lane and right_lane:    # we have both lanes
            left_lane_IPM = self.utils.get_line_IPM(left_lane)
            right_lane_IPM = self.utils.get_line_IPM(right_lane)
            pass
        else:
            if left_lane is not None:   # only have our left lane
                left_lane_IPM = self.utils.get_line_IPM(left_lane)
                right_lane_IPM = self.utils.translation_IPM(left_lane_IPM, True)
                right_lane = self.utils.get_inv_line_IPM(right_lane_IPM)
                y1_cv, x1_cv, y2_cv, x2_cv = right_lane
                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)
            else:
                if right_lane is not None:  # only have our right lane
                    right_lane_IPM = self.utils.get_line_IPM(right_lane)
                    left_lane_IPM = self.utils.translation_IPM(right_lane_IPM, False)
                    left_lane = self.utils.get_inv_line_IPM(left_lane_IPM)
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
        cv2.line(frame_ROI_IPM, (y_heading_road_cv, x_heading_road_cv), (y_bottom_road_cv, x_bottom_road_cv), (255, 255, 255), 3)
        road_line_reference = self.utils.get_inv_line_IPM([y_heading_road_cv, x_heading_road_cv, self.y_heading_car_cv, self.height_ROI_IPM])
        y1_cv, x1_cv, y2_cv, x2_cv = road_line_reference
        cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 255, 255), 3)

        theta = round(math.degrees(math.atan((y_heading_road_cv - self.y_heading_car_cv) / (x_heading_road_cv - self.height_ROI_IPM))))

        # offset
        offset = (y_bottom_road_cv - self.y_heading_car_cv) * self.pixel_resolution

        return offset, theta

    def optimized_detection(self, frame_ROI_preprocessed, frame_ROI, frame_ROI_IPM):
        return None, None


    def lane_detection(self, frame_ROI, frame_ROI_IPM=None):
        """

        :param frame_ROI_IPM: only used when we want draw some results on our IPM frame
        :return: the coordinates of left and right lanes of the road
        """
        frame_ROI_preprocessed = self.preprocessing(frame_ROI)
        # cv2.imshow("ROI Preprocessed", frame_ROI_preprocessed)
        # check for history of detected road lanes
        left_lane = None
        right_lane = None
        if self.previous_left_lane and self.previous_right_lane:
            # TO DO: implement algorithm using data from previous frames for optimization
            left_lane, right_lane = self.optimized_detection(frame_ROI_preprocessed, frame_ROI, frame_ROI_IPM)
            pass
        else:
            left_lane, right_lane = self.first_detection(frame_ROI_preprocessed, frame_ROI, frame_ROI_IPM)
        offset = None
        theta = None
        if left_lane is not None or right_lane is not None:
            offset, theta = self.get_offset_theta(frame_ROI, left_lane, right_lane, frame_ROI_IPM)
        else:
            # algorithm for estimating lanes when they are not detected in our current frame
            # for example: Kalman Filter
            pass
        return offset, theta


    def run(self):
        ret, frame = self.cap.read()
        while True:
            start = time.time()

            # select our ROI
            frame_ROI = frame[self.x_cv_ROI:, :]
            frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM),
                                                flags=cv2.INTER_NEAREST)

            offset, theta = self.lane_detection(frame_ROI, frame_ROI_IPM=frame_ROI_IPM)

            if offset is not None:
                print("offset = {} cm".format(offset))
            if theta is not None:
                print("theta = {}".format(theta))

            # cv2.imshow("Frame", frame)
            # cv2.imshow("IPM", frame_ROI_IPM)
            cv2.imshow("ROI", frame_ROI)
            cv2.waitKey(1)

            end = time.time()
            print("time: {}".format(end - start))

            _, frame = self.cap.read()


LD = LaneDetection()
LD.run()
