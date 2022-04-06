import numpy as np
import cv2
import math


class Utils:

    def __init__(self):

        ''' Info about frame'''
        self.width = 640
        self.height = 480

        ''' Info about ROI '''
        self.x_cv_ROI = 270
        self.height_ROI = self.height - self.x_cv_ROI
        self.width_ROI = self.width

    def draw_line(self, line, color, image):
        y1_cv, x1_cv, y2_cv, x2_cv = line[0]
        radius = 5
        color_left_most_point = (0, 255, 0)  # GREEN for left_most point
        color_right_most_point = (255, 0, 0)  # BLUE fpr right_most point
        cv2.circle(image, (y1_cv, x1_cv), radius, color_left_most_point, 1)
        cv2.circle(image, (y2_cv, x2_cv), radius, color_right_most_point, 1)
        cv2.line(image, (y1_cv, x1_cv), (y2_cv, x2_cv), color, 2)

    def linspace(self, line):
        x1, y1, x2, y2 = line
        num = 5
        points = []
        if x1 < x2:
            x_min = x1
            x_max = x2
        else:
            x_min = x2
            x_max = x1
        x_array = np.linspace(x_min, x_max, num)
        coeff = np.polynomial.polynomial.polyfit((x1, x2), (y1, y2), deg=1)
        for x in x_array:
            y = int(coeff[1] * x + coeff[0])
            points.append((int(x), y))
        return points

    def root(self, coefficient, derivative, left_lane=False):
        delta = math.sqrt(coefficient[1] ** 2 - 4 * coefficient[2] * coefficient[0])
        x1 = (-coefficient[1] - delta) // (2 * coefficient[2])
        x2 = (-coefficient[1] + delta) // (2 * coefficient[2])
        if left_lane:
            if derivative[1] * x1 + derivative[0] > 0:
                return int(x1)
            if derivative[1] * x2 + derivative[0] > 0:
                return int(x2)
        else:
            if derivative[1] * x1 + derivative[0] <= 0:
                return int(x1)
            if derivative[1] * x2 + derivative[0] <= 0:
                return int(x2)

    def polyfit(self, lines, frame_ROI, left_lane=False):
        # coordinates used for estimating our line
        x_points = []
        y_points = []

        for line in lines:
            x1, y1, x2, y2 = self.get_XoY_coordinates(line)
            x_points.append(x1)
            x_points.append(x2)
            y_points.append(y1)
            y_points.append(y2)
            # add more coordinates on the line for better precision in estimating our lane
            # ------------------------------------------------------------------------------
            num = 7
            if x1 < x2:
                x_min = x1
                x_max = x2
            else:
                x_min = x2
                x_max = x1
            x_array = np.linspace(x_min, x_max, num)
            coeff = np.polynomial.polynomial.polyfit((x1, x2), (y1, y2), deg=1)
            for x in x_array:
                y = int(coeff[1] * x + coeff[0])
                y_cv = int(x)
                x_cv = abs(y - self.height_ROI)
                cv2.circle(frame_ROI, (y_cv, x_cv), 5, (0, 0, 255), 1)
                x_points.append(x)
                y_points.append(y)
                # ---------------
                x_points.append(x-3)
                y_points.append(y)
                # ---------------
                x_points.append(x+3)
                y_points.append(y)
                # ---------------
                x_points.append(x-6)
                y_points.append(y)
                # ---------------
                x_points.append(x+6)
                y_points.append(y)
            # ------------------------------------------------------------------------------

        coefficient = np.polynomial.polynomial.polyfit(x_points, y_points, deg=1)
        # ----------------------------------------------------------------------
        # x_mean = np.mean(x_points)
        # y_mean = np.mean(y_points)
        # n = len(x_points)
        #
        # # Calculate the linear equation
        # numerator = 0  # top
        # denominator = 0  # bottom
        #
        # for (x, y) in zip(x_points, y_points):
        #     numerator += (x - x_mean) * (y - y_mean)
        #     denominator += (x - x_mean) ** 2
        #
        # slope = numerator / denominator
        # intercept_oy = y_mean - slope * x_mean
        #
        # print("y = {}*x + {}".format(slope, intercept_oy))

        # ----------------------------------------------------------------------
        # expand our estimated line from bottom to the top of the ROI
        y1 = 0
        y2 = self.height_ROI
        x1 = int((y1 - coefficient[0]) / coefficient[1])
        x2 = int((y2 - coefficient[0]) / coefficient[1])

        # convert our estimated line from XoY in cv2 coordinate system
        y1_cv, x1_cv, y2_cv, x2_cv = self.get_cv2_coordinates([x1, y1, x2, y2])
        cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)
        return [y1_cv, x1_cv, y2_cv, x2_cv]


    def left_or_right_candidate_line(self, intercept_oX, theta):  # 0 -> left line;   # 1 -> right line;
        y_cv_margin = 145  # offset wrt to the center vertical line
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
        return -1  # no line of the road

    def get_homography_matrix(self, src_points, dst_points, pixel_resolution):
        """

        :return: Homography matrix use for IPM
        """
        dst_points = np.array(
            [[int(y_cv / pixel_resolution), int(x_cv / pixel_resolution)] for [y_cv, x_cv] in dst_points],
            dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        return H

    def get_intercept_theta(self, line):
        """
        We get the intercept with real Ox axis of and the slope of our line expressed in degrees
        """
        x1, y1, x2, y2 = self.get_XoY_coordinates(line)
        coefficients = np.polynomial.polynomial.polyfit((x1, x2), (y1, y2), deg=1)
        if coefficients is not None:
            # coefficients[1] = slope
            # coefficients[0] = intercept on oY
            # y = slope * x + intercept_oY
            theta = math.degrees(math.atan(coefficients[1]))
            # TO DO: handle vertical and horizontal lines!
            try:
                intercept_oX = int((-coefficients[0]) / coefficients[1])
            except OverflowError:
                intercept_oX = 30000

            return intercept_oX, theta

    def get_cv2_coordinates(self, line):
        x1, y1, x2, y2 = line
        return x1, abs(y1 - self.height_ROI),  x2, abs(y2 - self.height_ROI)

    def get_XoY_coordinates(self, line):
        """
        Transform the cv2 coordinates into real xOy coordinates
        """
        y1_cv, x1_cv, y2_cv, x2_cv = line[0]
        return y1_cv, abs(x1_cv - self.height_ROI), y2_cv, abs(x2_cv - self.height_ROI)
