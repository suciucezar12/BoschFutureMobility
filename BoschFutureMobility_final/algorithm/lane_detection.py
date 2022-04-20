import time
import math
import cv2
import numpy as np
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
        self.x_cv_ROI = 320
        self.height_ROI = self.height - self.x_cv_ROI
        self.width_ROI = self.width

        ''' Info for IPM (Inverse Perspective Mapping)'''
        self.src_points_DLT = np.array(
            [[0, 0], [self.width_ROI, 0], [self.width_ROI, self.height_ROI], [0, self.height_ROI]],
            dtype=np.float32)
        self.dst_points_DLT = np.array([[0, 2.2], [57.7, 0], [49.5, 13.2], [7.5, 14.2]])  # expressed in centimeters
        self.pixel_resolution = float(57.7 / self.width_ROI)  # centimeters per pixel
        self.H = self.utils.get_homography_matrix(self.src_points_DLT, self.dst_points_DLT,
                                                  self.pixel_resolution)
        self.inv_H = np.linalg.inv(self.H)
        self.height_ROI_IPM = 147  # calculated related to pixel_resolution and the real dimensions
        self.width_ROI_IPM = 640

        ''' Centroid of front axel '''
        # TO DO: modify this values
        self.offset_origin = -20  # to correct the inclination of our camera
        self.y_heading_car_cv = self.width_ROI_IPM // 2 + self.offset_origin
        self.width_road_IPM = 310

    def preprocessing(self, frame_ROI):
        grayscale_frame = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
        contrast_frame = cv2.convertScaleAbs(grayscale_frame, alpha=1.5, beta=0)
        # cv2.imshow("Contrast_ROI", contrast_frame)
        canny_frame = cv2.Canny(contrast_frame, 150, 200)
        return canny_frame

    def detect_lanes(self, frame_ROI_preprocessed, frame_ROI, frame_ROI_IPM):
        lines_candidate = cv2.HoughLinesP(frame_ROI_preprocessed, rho=1, theta=np.pi / 180, threshold=45,
                                          minLineLength=20,
                                          maxLineGap=80)

        if lines_candidate is not None:
            left_lines, right_lines, horizontal_lines = self.utils.filter_lines(lines_candidate, frame_ROI, frame_ROI_IPM)
            left_lane = self.utils.estimate_lane(left_lines, self.height_ROI, frame_ROI, frame_ROI_IPM)
            right_lane = self.utils.estimate_lane(right_lines, self.height_ROI, frame_ROI, frame_ROI_IPM)
            return left_lane, right_lane
        else:
            return None, None

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
                # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)
            else:
                if right_lane is not None:  # only have our right lane
                    right_lane_IPM = self.utils.get_line_IPM(right_lane, self.H)
                    left_lane_IPM = self.utils.translation_IPM(right_lane_IPM, self.width_road_IPM, False)
                    left_lane = self.utils.get_line_IPM(left_lane_IPM, self.inv_H)
                    y1_cv, x1_cv, y2_cv, x2_cv = left_lane
                    # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)

        if left_lane_IPM is not None and right_lane_IPM is not None:
            y1_left_cv, x1_left_cv, y2_left_cv, x2_left_cv = left_lane_IPM
            y1_right_cv, x1_right_cv, y2_right_cv, x2_right_cv = right_lane_IPM
            # if frame_ROI_IPM is not None:
                # cv2.line(frame_ROI_IPM, (y1_left_cv, x1_left_cv), (y2_left_cv, x2_left_cv), (0, 255, 0), 3)
                # cv2.line(frame_ROI_IPM, (y1_right_cv, x1_right_cv), (y2_right_cv, x2_right_cv), (0, 255, 0), 3)

            # theta
            y_heading_road_cv = (y2_left_cv + y2_right_cv) // 2
            x_heading_road_cv = (x2_left_cv + x2_right_cv) // 2

            y_bottom_road_cv = (y1_left_cv + y1_right_cv) // 2
            x_bottom_road_cv = (x1_left_cv + x1_right_cv) // 2
            # cv2.line(frame_ROI_IPM, (y_heading_road_cv, x_heading_road_cv), (y_bottom_road_cv, x_bottom_road_cv),
            #          (255, 255, 255), 3)
            road_line_reference = self.utils.get_line_IPM(
                [y_heading_road_cv, x_heading_road_cv, self.y_heading_car_cv, self.height_ROI_IPM], self.inv_H)
            y1_cv, x1_cv, y2_cv, x2_cv = road_line_reference
            self.road_lane = road_line_reference
            # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 255, 255), 3)

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

    def lane_detection(self, frame_ROI, frame_ROI_IPM):
        frame_ROI_preprocessed = self.preprocessing(frame_ROI)
        cv2.imshow("ROI_Preprocessed", frame_ROI_preprocessed)

        if self.previous_left_lane and self.previous_right_lane:
            pass
            # # TO DO: implement algorithm using data from previous frames for optimization
            # left_lane, right_lane = self.optimized_detection(frame_ROI, frame_ROI_IPM)
            # offset, theta, left_lane_IPM, right_lane_IPM = self.get_offset_theta(frame_ROI, left_lane, right_lane,
            #                                                                      frame_ROI_IPM)
            # intersection = self.optimized_intersection_detection(frame_ROI, left_lane_IPM, right_lane_IPM,
            #                                                      frame_ROI_IPM)
        else:
            left_lane, right_lane = self.detect_lanes(frame_ROI_preprocessed, frame_ROI, frame_ROI_IPM)
            offset, theta, left_lane_IPM, right_lane_IPM = self.get_offset_theta(frame_ROI, left_lane, right_lane,
                                                                                 frame_ROI_IPM)
            return theta, offset
            # if left_lane_IPM is not None and right_lane_IPM is not None:
                # self.left_lane = left_lane
                # self.right_lane = right_lane
                # if len(horizontal_lines):
                #     intersection = self.intersection_detection(frame_ROI, horizontal_lines, left_lane_IPM,
                #                                                right_lane_IPM,
                #                                                frame_ROI_IPM)
        return None, None



    def run(self):

        ret, frame = self.cap.read()

        while True:
            start = time.time()
            frame_ROI = frame[self.x_cv_ROI:, :]
            # frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM),
            #                                     flags=cv2.INTER_NEAREST)

            theta, offset = self.lane_detection(frame_ROI, None)

            if offset is not None:
                print("OFFSET = {} cm".format(offset))
                # offset_prev = offset

            if theta is not None:
                print("THETA = {}".format(theta))
                # theta_prev = theta

            cv2.imshow("Frame", frame)
            # cv2.imshow("ROI", frame_ROI)
            # cv2.imshow("IPM", frame_ROI_IPM)
            end = time.time()
            print("time = {}".format(end - start))
            cv2.waitKey(1)
            _, frame = self.cap.read()


LD = LaneDetection()
LD.run()
