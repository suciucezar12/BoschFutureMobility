import time

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

    def preprocessing(self, frame_ROI):
        grayscale_frame = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
        contrast_frame = cv2.convertScaleAbs(grayscale_frame, alpha=1.5, beta=0)
        # cv2.imshow("Contrast_ROI", contrast_frame)
        canny_frame = cv2.Canny(contrast_frame, 150, 200)
        return canny_frame

    def filter_lines(self, lines_candidate, frame_ROI, frame_ROI_IPM):
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
            if y1_cv != y2_cv:
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
                                # self.left_lines.append(line)
                                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 0, 0), 2)

                            if margin_y_cv_right <= intercept_oX <= self.width_ROI:  # right line
                                right_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                # self.right_lines.append(line)
                                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 0, 255), 2)

                            # check by theta and intercept_oX (last criteria)
                            if coeff[1] <= -0.2:  # candidate left line
                                if 0 <= intercept_oX <= margin_y_cv_right:
                                    left_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                    # self.left_lines.append(line)
                                    cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 0, 0), 2)

                            if coeff[1] >= 0.2:  # candidate right line
                                if margin_y_cv_left <= intercept_oX <= self.width_ROI:
                                    right_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                    # self.right_lines.append(line)
                                    cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 0, 255), 2)
                        else:
                            if abs(coeff[1]) <= 0.3:
                                horizontal_lines.append(line)
                                # self.horizontal_lines.append(line)

        return left_lines, right_lines, horizontal_lines

    def detect_lanes(self, frame_ROI_preprocessed, frame_ROI, frame_ROI_IPM):
        lines_candidate = cv2.HoughLinesP(frame_ROI_preprocessed, rho=1, theta=np.pi / 180, threshold=45,
                                          minLineLength=20,
                                          maxLineGap=30)

        if lines_candidate is not None:
            left_lines, right_lines, horizontal_lines = self.filter_lines(lines_candidate, frame_ROI, frame_ROI_IPM)



    def lane_detection(self, frame_ROI, frame_ROI_IPM):
        frame_ROI_preprocessed = self.preprocessing(frame_ROI)
        cv2.imshow("ROI_Preprocessed", frame_ROI_preprocessed)

        # get left and right lane
        self.detect_lanes(frame_ROI_preprocessed, frame_ROI, frame_ROI_IPM)



    def run(self):

        ret, frame = self.cap.read()

        while True:
            start = time.time()
            frame_ROI = frame[self.x_cv_ROI:, :]
            # frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM),
            #                                     flags=cv2.INTER_NEAREST)

            self.lane_detection(frame_ROI, None)

            cv2.imshow("Frame", frame)
            # cv2.imshow("ROI", frame_ROI)
            # cv2.imshow("IPM", frame_ROI_IPM)
            end = time.time()
            print("time = {}".format(end - start))
            cv2.waitKey(1)
            _, frame = self.cap.read()


LD = LaneDetection()
LD.run()
