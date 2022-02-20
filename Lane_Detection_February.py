import time
import numpy as np
import cv2

class LaneDetection:

    def __init__(self):
        ''' Matrix used for IPM '''
        self.x_top = 270    # Coordinates of the polygon we use for creating the Homography matrix
        self.y_left_top = 80
        self.y_right_top = 560
        self.input_coordinates_IPM = np.array([[0, 480], [self.y_left_top, self.x_top], [self.y_right_top, self.x_top], [640, 480]], dtype=np.float32)
        self.output_coordinates_IPM = np.array([[199, 36], [417, 0], [439, 444], [205, 410]], dtype=np.float32) # Output coordinates calculated manually in our flat real word plane of the road
        self.matrix_IPM = cv2.getPerspectiveTransform(self.input_coordinates_IPM, self.output_coordinates_IPM)
        ''' =========================================== '''

        time.sleep(1)
        self.cap = cv2.VideoCapture(0)

    def get_IPM_frame(self, frame):
        frame_IPM_width = 450
        frame_IPM_height = 300
        frame_IPM = cv2.warpPerspective(frame, self.matrix_IPM, (frame_IPM_width, frame_IPM_height), flags=cv2.INTER_LINEAR)
        rotation_matrix = cv2.getRotationMatrix2D((frame_IPM_width / 2, frame_IPM_height / 2), 90, 1.0)
        frame_IPM_rotated = cv2.warpAffine(frame_IPM, rotation_matrix, (frame_IPM_width, frame_IPM_height))
        return frame_IPM_rotated

    def run(self):

        ret, frame = self.cap.read()

        while True:

            start = time.time()  # measure (time_computing / frame)

            cv2.polylines(frame, np.int32([self.input_coordinates_IPM]), True, (0, 255, 255))
            frame_IPM = self.get_IPM_frame(frame)

            cv2.imshow("IPM", frame_IPM)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

            end = time.time()
            print(end - start)

            ret, frame = self.cap.read()



LD = LaneDetection()
LD.run()
