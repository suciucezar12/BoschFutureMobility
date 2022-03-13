import math
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

    def get_XoY_coordinates(self, line):
        # cv2 coordinates
        y1_cv, x1_cv, y2_cv, x2_cv = line[0]
        # conversion to usual XoY coordinate system
        # x1 = y1_cv
        # x2 = y2_cv
        # y1 = abs(x1_cv - self.height_ROI)
        # y2 = abs(x2_cv - self.height_ROI)
        return y1_cv, abs(x1_cv - self.height_ROI), y2_cv, abs(x2_cv - self.height_ROI)

    def get_intercept_theta_line(self, line):
        # conversion to usual XoY coordinate system
        x1, y1, x2, y2 = self.get_XoY_coordinates(line)
        coefficients = np.polynomial.polynomial.polyfit((x1, x2), (y1, y2), 1)
        if coefficients is not None:
            # coefficients[1] = slope
            # coefficients[0] = intercept on oY
            # y = slope * x + intercept_oY
            theta = math.degrees(math.atan(coefficients[1]))
            try:    # huge values of intercept_oY because of horizontal lines
                intercept_oX = int((-coefficients[0]) / coefficients[1])
            except OverflowError:
                intercept_oX = 30000

            return intercept_oX, theta

    def left_or_right_candidate_line(self, intercept_oX, theta):    # 0 -> left line;   # 1 -> right line;
        y_cv_margin = 145   # offset wrt to the center vertical line
        margin_y_cv_left = int(self.width_ROI / 2) - y_cv_margin
        margin_y_cv_right = int(self.width_ROI / 2) + y_cv_margin
        # intercept location has higher priority
        if -50 <= intercept_oX <= margin_y_cv_left:  # left line
            return 0
        if margin_y_cv_right <= intercept_oX <= self.width_ROI + 50:  # right line
            return 1
        # check by theta and intercept_oX (last criteria)
        if theta > 0:  # candidate left line
            if -50 <= intercept_oX <= margin_y_cv_right:
                return 0
        if theta < 0:  # candidate right line
            if margin_y_cv_left <= intercept_oX <= self.width_ROI + 50:
                return 1
        return -1   # no line of the road

    def filter_lines(self, lines_candidate, frame_ROI, frame_ROI_IPM=None):
        horizontal_lines = []
        left_lines = []
        right_lines = []
        for line in lines_candidate:
            intercept_oX, theta = self.get_intercept_theta_line(line)
            # horizontal line
            if abs(theta) <= 35:
                horizontal_lines.append(line)
            else:
                # left/right lane
                line_code = self.left_or_right_candidate_line(intercept_oX, theta)
                if line_code == 0:  # left line
                    self.draw_line(line, (255, 0, 0), frame_ROI)    # BLUE = LEFT
                    left_lines.append(line)
                if line_code == 1:  # right line
                    self.draw_line(line, (0, 0, 255), frame_ROI)    # RED = RIGHT
                    right_lines.append(line)

    def get_left_and_right_line(self, frame_ROI, frame_ROI_IPM=None):   # get left and right lines of the road
        frame_ROI_preprocessed = self.preprocess(frame_ROI)
        # detected possible lines of our road
        lines_candidate = cv2.HoughLinesP(frame_ROI_preprocessed, rho=1, theta=np.pi / 180, threshold=50, minLineLength=35,
                                maxLineGap=80)
        # filter lines which are not candidate for road's lanes
        if lines_candidate is not None:
            self.filter_lines(lines_candidate, frame_ROI, frame_ROI_IPM)

    def get_theta(self, frame_ROI, frame_ROI_IPM=None):  # get the steering angle
        self.get_left_and_right_line(frame_ROI, frame_ROI_IPM)

    def run(self):
        ret, frame = self.cap.read()

        while True:
            start = time.time()
            frame_ROI = frame[self.x_cv_ROI:, :]
            # frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM), flags=cv2.INTER_LINEAR)

            self.get_theta(frame_ROI)
            print("time: {}".format(time.time() - start))

            cv2.imshow("Frame", frame)
            cv2.imshow("ROI", frame_ROI)
            # cv2.imshow("IPM", frame_ROI_IPM)
            cv2.waitKey(1)

            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
