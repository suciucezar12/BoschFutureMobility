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
        self.H = self.get_homography_matrix()   # Homography Matrix for IPM

        # size of ROI_IPM
        self.height_ROI_IPM = 210  # calculated related to pixel_resolution and the real dimensions
        self.width_ROI_IPM = 547

    def get_homography_matrix(self):
        src_points = np.array([[0, 0], [self.width_ROI, 0], [self.width_ROI, self.height_ROI], [0, self.height_ROI]],
                              dtype=np.float32)
        dst_points = np.array([[0, 3.1], [66.7, 0], [53.5, 24.6], [11, 25.6]])
        dst_points = np.array(
            [[int(y_cv / self.pixel_resolution), int(x_cv / self.pixel_resolution)] for [y_cv, x_cv] in dst_points],
            dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_points, dst_points)  # Homography matrix for IPM
        return H

    def preprocess(self, frame_ROI):    # preprocessing phase of our pipeline
        frame_ROI_gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
        frame_ROI_blurred = cv2.GaussianBlur(frame_ROI_gray, (11, 11), 0)
        frame_ROI_preprocessed = cv2.Canny(frame_ROI_blurred, 30, 255)
        return frame_ROI_preprocessed

    def draw_line(self, line, color, image):
        y1_cv, x1_cv, y2_cv, x2_cv = line[0]
        radius = 5
        color_left_most_point = (0, 255, 0)  # GREEN for left_most point
        color_right_most_point = (255, 0, 0)  # BLUE fpr right_most point
        cv2.circle(image, (y1_cv, x1_cv), radius, color_left_most_point, 1)
        cv2.circle(image, (y2_cv, x2_cv), radius, color_right_most_point, 1)
        cv2.line(image, (y1_cv, x1_cv), (y2_cv, x2_cv), color, 2)

    def get_left_and_right_lines(self, frame_ROI, frame_ROI_IPM=None):   # get left and right lines of the road
        frame_ROI_preprocessed = self.preprocess(frame_ROI)
        # detected possible lines of our road
        lines_candidate = cv2.HoughLinesP(frame_ROI_preprocessed, rho=1, theta=np.pi / 180, threshold=50, minLineLength=20,
                                maxLineGap=80)
        if lines_candidate is not None:
            for line in lines_candidate:
                self.draw_line(line, (0, 0, 255), frame_ROI)

    def get_theta(self, frame_ROI, frame_ROI_IPM=None):  # get the steering angl
        self.get_left_and_right_lines(frame_ROI, frame_ROI_IPM)

    def run(self):
        ret, frame = self.cap.read()

        while True:
            start = time.time()
            frame_ROI = frame[self.x_cv_ROI:, :]
            # frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM), flags=cv2.INTER_LINEAR)

            self.get_theta(frame_ROI)
            print("time: {}".format(time.time() - start))

            cv2.imshow("Frame", frame)
            # cv2.imshow("ROI", frame_ROI)
            # cv2.imshow("IPM", frame_ROI_IPM)
            cv2.waitKey(1)

            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
