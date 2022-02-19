import time

import cv2

class LaneDetection:

    def __init__(self):
        ''' Declare extrinsic and intrinsic parameters '''
        ''' ========================================== '''

        ''' The coordinates of Region of Interest (ROI) '''
        self.x_top = 300
        self.y_left_top = 50
        self.y_right_top = 590
        self.roi_coords = [[480, 0], [self.x_top, self.y_left_top], [self.x_top, self.y_right_top], [[480, 640]]]
        ''' =========================================== '''

        time.sleep(1)
        self.cap = cv2.VideoCapture(0)

    def run(self):

        ret, frame = self.cap.read()

        while True:

            cv2.polylines(frame, self.roi_coords, True, (0,255,255))

            ret, frame = self.cap.read()
