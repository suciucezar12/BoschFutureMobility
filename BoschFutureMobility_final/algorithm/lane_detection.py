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
        self.height_ROI_IPM = 147  # calculated related to pixel_resolution and the real dimensions
        self.width_ROI_IPM = 640

        self.offset_origin = -20  # to correct the inclination of our camera
        self.y_heading_car_cv = self.width_ROI_IPM // 2 + self.offset_origin

    def run(self):

        ret, frame = self.cap.read()

        while True:

            frame_ROI = frame[self.x_cv_ROI:, :]
            frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM),
                                                flags=cv2.INTER_NEAREST)

            cv2.imshow("ROI", frame_ROI)
            cv2.imshow("IPM", frame_ROI_IPM)

            _, frame = self.cap.read()


LD = LaneDetection()
LD.run()
