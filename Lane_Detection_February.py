import time
import numpy as np
import cv2

class LaneDetection:

    def __init__(self):
        ''' Declare extrinsic and intrinsic parameters '''
        ''' ========================================== '''

        ''' The coordinates of Region of Interest (ROI) '''
        self.x_top = 300
        self.y_left_top = 130
        self.y_right_top = 510
        # self.roi_coords = np.array([[480, 0], [self.x_top, self.y_left_top], [self.x_top, self.y_right_top], [480, 640]], dtype=np.int32)
        # coords are [y, x]
        self.roi_coords = np.array([[0, 480], [self.y_left_top, self.x_top], [self.y_right_top, self.x_top], [640, 480]], dtype=np.int32)
        ''' =========================================== '''

        time.sleep(1)
        self.cap = cv2.VideoCapture(0)

    def run(self):

        ret, frame = self.cap.read()

        while True:

            cv2.polylines(frame, [self.roi_coords], True, (0,255,255))
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
