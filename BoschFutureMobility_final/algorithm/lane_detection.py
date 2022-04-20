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
        self.pixel_resolution = 57.7 / self.width_ROI  # centimeters per pixel
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
        contrast_frame = cv2.convertScaleAbs(grayscale_frame, alpha=1.3, beta=0)
        # cv2.imshow("Contrast_ROI", contrast_frame)
        canny_frame = cv2.Canny(contrast_frame, 150, 200)
        return canny_frame

    def lane_detection(self, frame_ROI, frame_ROI_IPM):
        frame_ROI_preprocessed = self.preprocessing(frame_ROI)
        cv2.imshow("ROI_Preprocessed", frame_ROI_preprocessed)

    def run(self):

        ret, frame = self.cap.read()

        while True:
            start = time.time()
            frame_ROI = frame[self.x_cv_ROI:, :]
            frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM),
                                                flags=cv2.INTER_NEAREST)

            self.lane_detection(frame_ROI, None)

            cv2.imshow("ROI", frame_ROI)
            cv2.imshow("IPM", frame_ROI_IPM)
            cv2.waitKey(1)
            end = time.time()
            print("time = {}".format(end - start))
            _, frame = self.cap.read()


LD = LaneDetection()
LD.run()
