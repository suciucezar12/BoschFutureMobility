import time

import cv2
import numpy as np
import utils


class LaneDetection:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        ''' Info about frame'''
        self.width = 640
        self.height = 480

        ''' Info about ROI '''
        self.x_cv_ROI = 270
        self.height_ROI = self.width - self.x_cv_ROI
        self.width_ROI = self.width

        ''' Info for IPM (Inverse Perspective Mapping)'''
        self.src_points_DLT = np.array(
            [[0, 0], [self.width_ROI, 0], [self.width_ROI, self.height_ROI], [0, self.height_ROI]],
            dtype=np.float32)
        self.dst_points_DLT = np.array([[0, 3.1], [66.7, 0], [53.5, 24.6], [11, 25.6]])  # expressed in centimeters
        self.pixel_resolution = 0.122  # centimeters per pixel
        self.H = utils.get_homography_matrix(self.src_points_DLT, self.dst_points_DLT,
                                             self.pixel_resolution)

    def preprocessing(self, frame_ROI):
        frame_ROI_gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
        frame_ROI_blurred = cv2.GaussianBlur(frame_ROI_gray, (5, 5), 0)
        frame_ROI_canny = cv2.Canny(frame_ROI_blurred, 180, 255)
        return frame_ROI_canny

    def run(self):
        ret, frame = self.cap.read()
        while True:
            start = time.time()
            # select our ROI
            frame_ROI = frame[self.x_cv_ROI:, :]
            frame_ROI_canny = self.preprocessing(frame_ROI)
            # cv2.imshow("Frame", frame)
            cv2.imshow("ROI Preprocessed", frame_ROI_canny)
            cv2.waitKey(1)
            end = time.time()
            print("time: {}".format(end - start))
            _, frame = self.cap.read()


LD = LaneDetection()
LD.run()
