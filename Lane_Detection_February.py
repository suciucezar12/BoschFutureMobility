import time
import numpy as np
import cv2

class LaneDetection:

    def __init__(self):
        ''' Declare extrinsic and intrinsic parameters '''
        ''' ========================================== '''
        self.P = self.get_Homography_Matrix()

        ''' Matrix used for IPM '''
        self.x_top = 270    # Coordinates of the polygon we use for creating the Homography matrix
        self.y_left_top = 80
        self.y_right_top = 560
        self.input_coordinates_IPM = np.array([[0, 480], [self.y_left_top, self.x_top], [self.y_right_top, self.x_top], [640, 480]], dtype=np.float32)
        self.output_coordinates_IPM = np.array([[199, 36], [417, 0], [439, 444], [205, 410]], dtype=np.float32) # Output coordinates calculated manually in our flat real word plane of the road
        self.matrix_IPM = cv2.getPerspectiveTransform(self.input_coordinates_IPM, self.output_coordinates_IPM)

        ''' The coordinates of Region of Interest (ROI) '''
        # self.x_top = 270
        # self.y_left_top = 80
        # self.y_right_top = 560
        # # coords are [y, x]
        # self.roi_coords = np.array([[0, 480], [self.y_left_top, self.x_top], [self.y_right_top, self.x_top], [640, 480]], dtype=np.int32)
        # self.roi_coords = np.array(
        #     [[0, 480], [0, self.x_top], [640, self.x_top], [640, 480]], dtype=np.int32)
        ''' =========================================== '''

        time.sleep(1)
        self.cap = cv2.VideoCapture(0)

    def get_IPM_frame(self, frame):
        frame_IPM = cv2.warpPerspective(frame, self.matrix_IPM, (600, 600), flags=cv2.INTER_LINEAR)
        rotation_matrix = cv2.getRotationMatrix2D((300, 300), 90, 1.0)
        frame_IPM_rotated = cv2.warpAffine(frame_IPM, rotation_matrix, (600, 600))
        return frame_IPM_rotated


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
        print(P)
        return P


    def run(self):

        ret, frame = self.cap.read()

        while True:

            start = time.time()

            cv2.polylines(frame, np.int32([self.input_coordinates_IPM]), True, (0,255,255))

            # out = cv2.warpPerspective(frame, self.P, (640, 480 - self.x_top), flags=cv2.INTER_LINEAR)

            # coordinates correspondents
            # output_pts = np.float32([[199, 36],
            #                          [417, 0],
            #                          [439, 444],
            #                          [205, 410]])

            # output_pts = np.float32([[199, 107],
            #                          [417, 0],
            #                          [450, 645],
            #                          [483, 481]])

            # M = cv2.getPerspectiveTransform(np.array(self.roi_coords, dtype=np.float32), output_pts)
            #
            # out = cv2.warpPerspective(frame, M, (500, 500), flags=cv2.INTER_LINEAR)
            #
            rotated = self.get_IPM_frame(frame)

            cv2.imshow("IPM", rotated)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

            end = time.time()
            print(end - start)
            ret, frame = self.cap.read()



LD = LaneDetection()
LD.run()
