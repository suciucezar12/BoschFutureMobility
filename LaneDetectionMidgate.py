import cv2
import numpy as np
import time

class LaneDetection:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        ''' :variables for ROI and IPM '''
        self.x_cv_ROI = 270
        self.height_ROI = 210   # 480(frame.height) - 270
        self.width_ROI = 640
        self.pixel_resolution = 0.122  # centimeters per pixel
        self.H = self.get_Homography_Matrix()   # Homography Matrix for IPM

        # size of ROI_IPM
        self.height_ROI_IPM = 210  # calculated related to pixel_resolution and the real dimensions
        self.width_ROI_IPM = 547

    def get_Homography_Matrix(self):
        src_points = np.array([[0, 0], [self.width_ROI, 0], [self.width_ROI, self.height_ROI], [0, self.height_ROI]],
                              dtype=np.float32)
        dst_points = np.array([[0, 3.1], [66.7, 0], [53.5, 24.6], [11, 25.6]])
        dst_points = np.array(
            [[int(y_cv / self.pixel_resolution), int(x_cv / self.pixel_resolution)] for [y_cv, x_cv] in dst_points],
            dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_points, dst_points)  # Homography matrix for IPM
        return H

    def run(self):
        ret, frame = self.cap.read()

        while True:
            frame_ROI = frame[self.x_cv_ROI:, :]
            frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM), flags=cv2.INTER_LINEAR)

            cv2.imshow("Frame", frame)
            cv2.imshow("ROI", frame_ROI)
            cv2.imshow("IPM", frame_ROI_IPM)
            cv2.waitKey(1)

            ret, frame = self.cap.read()

LD = LaneDetection()
LD.run()