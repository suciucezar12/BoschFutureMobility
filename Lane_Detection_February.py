import time
import numpy as np
import cv2

class LaneDetection:

    def __init__(self):
        ''' Declare extrinsic and intrinsic parameters '''
        ''' ========================================== '''
        self.P = self.get_Homography_Matrix()

        ''' The coordinates of Region of Interest (ROI) '''
        self.x_top = 270
        self.y_left_top = 80
        self.y_right_top = 560
        # coords are [y, x]
        self.roi_coords = np.array([[0, 480], [self.y_left_top, self.x_top], [self.y_right_top, self.x_top], [640, 480]], dtype=np.int32)
        ''' =========================================== '''

        time.sleep(1)
        self.cap = cv2.VideoCapture(0)

    def get_Homography_Matrix(self):
        K = np.array([[530.59817269, 0., 315.86549494],
                      [0., 506.50082419, 238.34556175],
                      [0.,        0.,           1.]])

        R = np.array([[0., -1., 0.],
                      [0., 0., -1.],
                      [1., 0., 0.]])

        R_world2camera = np.array([[-0.95558857, 0.24619517, -0.16198279],
                      [-0.27596785, -0.55468898, 0.78495979],
                      [0.10340324, 0.79480065, 0.59799641]])

        T = np.array([[1., 0., 0.0],
                      [0., 1., 0.0],
                      [0., 0., 343.01101321]])

        RT = np.zeros((3,3))

        # RT[:, 0:2] = R_world2camera[:, 0:2]  # Z = 0
        # RT[:, 2] = T[:, 2]

        RT = R @ R_world2camera @ T

        P = K @ RT
        return P

    def run(self):

        ret, frame = self.cap.read()

        while True:

            cv2.polylines(frame, [self.roi_coords], True, (0,255,255))

            out = cv2.warpPerspective(frame, self.P, (640, 480 - self.x_top), flags=cv2.INTER_LINEAR)

            cv2.imshow("IPM", out)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
