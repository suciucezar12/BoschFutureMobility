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
        coefficients = np.polynomial.polynomial.polyfit((x1, x2), (y1, y2), deg=1)
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
                self.draw_line(line, (50, 50, 50), frame_ROI)
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

        return left_lines, right_lines, horizontal_lines

    def polyfit(self, lines, frame_ROI):    # polyfit on a set of coordinates of lines
        # coordinates used for estimating our line
        x_points = []
        y_points = []

        for line in lines:
            x1, y1, x2, y2 = self.get_XoY_coordinates(line)
            x_points.append(x1)
            x_points.append(x2)
            y_points.append(y1)
            y_points.append(y2)

        # get our estimated line
        coefficient = np.polynomial.polynomial.polyfit(x_points, y_points, deg=1)
        # print(str(coefficient[1]) + "*x + " + str(coefficient[0]))

        # expand our estimated line from bottom to the top of the ROI
        y1 = 0
        y2 = self.height_ROI
        x1 = int((y1 - coefficient[0]) / coefficient[1])
        x2 = int((y2 - coefficient[0]) / coefficient[1])

        # convert our estimated line from XoY in cv2 coordinate system
        y1_cv = x1
        y2_cv = x2
        x1_cv = abs(y1 - self.height_ROI)
        x2_cv = abs(y2 - self.height_ROI)

        cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)

        return y1_cv, x1_cv, y2_cv, x2_cv  # return the coordinates of our estimated line and its line equation

    def get_road_lines(self, frame_ROI, frame_ROI_IPM=None):   # get left and right lines of the road
        frame_ROI_preprocessed = self.preprocess(frame_ROI)
        # detected possible lines of our road
        lines_candidate = cv2.HoughLinesP(frame_ROI_preprocessed, rho=1, theta=np.pi / 180, threshold=50, minLineLength=35,
                                maxLineGap=80)
        # filter lines which are not candidate for road's lanes
        if lines_candidate is not None:
            left_lines, right_lines, horizontal_lines = self.filter_lines(lines_candidate, frame_ROI, frame_ROI_IPM)
            if len(left_lines) != 0 and len(right_lines) != 0:
                left_line = self.polyfit(left_lines, frame_ROI)
                right_line = self.polyfit(right_lines, frame_ROI)
            else:
                if len(left_lines) != 0:
                    left_line = self.polyfit(left_lines, frame_ROI)
                    right_line = None
                else:
                    if len(right_lines) != 0:
                        left_line = None
                        right_line = self.polyfit(right_lines, frame_ROI)
                    else:
                        left_line = None
                        right_line = None
            return left_line, right_line, horizontal_lines

    def get_line_IPM(self, line, image):
        y1_cv, x1_cv, y2_cv, x2_cv = line
        src_points = np.array([[[y1_cv, x1_cv], [y2_cv, x2_cv]]], dtype=np.float32)
        dest_points = cv2.perspectiveTransform(src_points, self.H)[0]
        print(dest_points[0][0])
        return dest_points[0][0], dest_points[0][1], dest_points[1][0], dest_points[1][1]

    def get_theta(self, frame_ROI, frame_ROI_IPM=None):  # get the steering angle
        left_line, right_line, horizontal_lines = self.get_road_lines(frame_ROI, frame_ROI_IPM)

        # testing IPM
        if left_line is not None:
            left_line_IPM = self.get_line_IPM(left_line, frame_ROI_IPM)
            self.draw_line(left_line_IPM, (0, 255, 0), frame_ROI_IPM)
        if right_line is not None:
            right_line_IPM = self.get_line_IPM(right_line, frame_ROI_IPM)
            self.draw_line(right_line_IPM, (0, 255, 0), frame_ROI_IPM)


    def run(self):
        ret, frame = self.cap.read()

        while True:
            start = time.time()
            frame_ROI = frame[self.x_cv_ROI:, :]
            frame_ROI_IPM = cv2.warpPerspective(frame_ROI, self.H, (self.width_ROI_IPM, self.height_ROI_IPM), flags=cv2.INTER_NEAREST)

            self.get_theta(frame_ROI, frame_ROI_IPM)
            # print("time: {}".format(time.time() - start))

            # cv2.imshow("Frame", frame)
            cv2.imshow("ROI", frame_ROI)
            cv2.imshow("IPM", frame_ROI_IPM)
            cv2.waitKey(1)

            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
