import math

import cv2
import numpy as np
import time

class LaneDetection:

    def __init__(self):
        ''' Matrix used for IPM '''
        self.width = 640
        self.height = 480
        self.x_top = 270  # Coordinates of the polygon we use for creating the Homography matrix
        self.y_left_top = 80
        self.y_right_top = 560
        self.input_coordinates_IPM = np.array(
            [[0, 480], [self.y_left_top, self.x_top], [self.y_right_top, self.x_top], [640, 480]], dtype=np.float32)
        self.output_coordinates_IPM = np.array([[199, 36], [417, 0], [439, 444], [205, 410]],
                                               dtype=np.float32)  # Output coordinates calculated manually in our flat real word plane of the road
        self.matrix_IPM = cv2.getPerspectiveTransform(self.input_coordinates_IPM, self.output_coordinates_IPM)
        ''' ================================================================================================================================ '''

        time.sleep(1)
        self.cap = cv2.VideoCapture(0)

    # preprocess our frame_ROI
    def preProcess(self, frame_ROI):
        frame_ROI_gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
        frame_ROI_blurred = cv2.GaussianBlur(frame_ROI_gray, (11, 11), 0)
        # cv2.imshow("ROI_Blurred", frame_ROI_blurred)

        frame_ROI_preprocessed = cv2.Canny(frame_ROI_blurred, 30, 255)

        return frame_ROI_preprocessed

    def get_intercept_theta_line(self, line, frame_ROI):
        height_ROI = frame_ROI.shape[0]
        # get cv2 coordinates of our line
        y1_cv, x1_cv, y2_cv, x2_cv = line[0]
        # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 0, 255), 2)

        # conversion to usual XoY coordinate system
        x1 = y1_cv
        x2 = y2_cv
        y1 = abs(x1_cv - height_ROI)
        y2 = abs(x2_cv - height_ROI)

        # get intercept and theta -> apply np.polyfit
        # y = slope * x + intercept_oY
        coefficients = np.polynomial.polynomial.polyfit((x1, x2), (y1, y2), 1)
        if coefficients is not None:
            # coefficients[1] = slope
            # coefficients[0] = intercept_oY
            theta = math.degrees(math.atan(coefficients[1]))
            # get intercept_oX  -> when y = 0;
            intercept_oX = int((-coefficients[0]) / coefficients[1])
            # print("theta = " + str(theta) + ";   intercept_oX = " + str(intercept_oX))
            return theta, intercept_oX

    def filter_line(self, theta, intercept_oX, width_ROI, y_cv_margin):

        margin_y_cv_left = int(width_ROI / 2) - y_cv_margin
        margin_y_cv_right = int(width_ROI / 2) + y_cv_margin

        if abs(theta) >= 35:    # if it's not horizontal
            # check by intercept_oX -> highest priority
            if -50 <= intercept_oX <= margin_y_cv_left:    # left line
                return 0
            if margin_y_cv_right <= intercept_oX <= width_ROI + 50:  # right line
                return 1
            # check by theta and intercept_oX
            if theta > 0:   # candidate left line
                if -50 <= intercept_oX <= margin_y_cv_right:
                    return 0
            if theta < 0:   # candidate right line
                if margin_y_cv_left <= intercept_oX <= width_ROI + 50:
                    return 1
        return -1

    def get_and_filter_lines(self, frame_ROI_preprocessed, frame_ROI):
        lines = cv2.HoughLinesP(frame_ROI_preprocessed, rho=1, theta=np.pi / 180, threshold=70, minLineLength=30,
                                         maxLineGap=70)
        left_lines = []
        right_lines = []

        width_ROI = frame_ROI.shape[1]
        y_cv_margin = 145  # margin wrt to vertical center of frame_ROI
        margin_y_cv_left = int(width_ROI / 2) + y_cv_margin
        margin_y_cv_right = int(width_ROI / 2) - y_cv_margin
        # draw lines for margin admitted
        cv2.line(frame_ROI, (int(width_ROI / 2) - y_cv_margin, 0), (int(width_ROI / 2) - y_cv_margin, frame_ROI.shape[0]), (0, 255, 0), 2)
        cv2.line(frame_ROI, (int(width_ROI / 2) + y_cv_margin, 0),
                 (int(width_ROI / 2) + y_cv_margin, frame_ROI.shape[0]), (0, 255, 0), 2)


        if lines is not None:
            for line in lines:
                theta, intercept_oX = self.get_intercept_theta_line(line, frame_ROI)
                line_code = self.filter_line(theta, intercept_oX, width_ROI, y_cv_margin)
                if line_code == 0:
                    y1_cv, x1_cv, y2_cv, x2_cv = line[0]
                    self.drawLane(line, frame_ROI, (0, 0, 255))
                    # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 0, 255), 2)     # RED color -> left_line
                    left_lines.append(line)
                if line_code == 1:
                    y1_cv, x1_cv, y2_cv, x2_cv = line[0]
                    self.drawLane(line, frame_ROI, (255, 0 ,0))
                    # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 0, 0), 2)     # BLUE color -> right_line
                    right_lines.append(line)

        return left_lines, right_lines

    def polyfit(self, lines, frame_ROI):
        # coordinates used for estimating our line
        x_points = []
        y_points = []

        for line in lines:
            y1_cv, x1_cv, y2_cv, x2_cv = line[0]    # coordinates in cv2 coordinate system
            # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), ())

            # conversion to usual XoY coordinate system
            x1 = y1_cv
            x2 = y2_cv
            y1 = abs(x1_cv - self.x_top)
            y2 = abs(x2_cv - self.x_top)

            x_points.append(x1)
            x_points.append(x2)
            y_points.append(y1)
            y_points.append(y2)

        # get our estimated line
        coefficient = np.polynomial.polynomial.polyfit(x_points, y_points, 1)
        # print(str(coefficient[1]) + "*x + " + str(coefficient[0]))

        # expand our estimated line from bottom to the top of the ROI
        y1 = 0
        y2 = self.x_top
        x1 = int((y1 - coefficient[0]) / coefficient[1])
        x2 = int((y2 - coefficient[0]) / coefficient[1])

        # convert our estimated line from XoY in cv2 coordinate system
        y1_cv = x1
        y2_cv = x2
        x1_cv = abs(y1 - self.x_top)
        x2_cv = abs(y2 - self.x_top)

        cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)

        return coefficient  # return the coordinates of our estimated line and its line equation

    def both_lines_detected(self, left_line_coefficients, right_line_coefficients, frame_ROI):
        height_ROI = frame_ROI.shape[0]
        x_cv_theta = 0   # the x_cv2 coordinate where we intersect -> wrt to ROI

        # transform in XoY coordinate
        y_theta = abs(x_cv_theta - height_ROI)
        x_left_theta = int((y_theta - left_line_coefficients[0]) / left_line_coefficients[1])
        x_right_theta = int((y_theta - right_line_coefficients[0]) / right_line_coefficients[1])

        # convert back to cv2 coordinate system
        y_cv_left_line = x_left_theta
        y_cv_right_line = x_right_theta

        cv2.line(frame_ROI, (y_cv_left_line, x_cv_theta), (y_cv_right_line, x_cv_theta), (200, 200, 200), 2)

        y_cv_vanishing_point = int((y_cv_right_line + y_cv_left_line) / 2)
        # print(y_cv_vanishing_point - y_cv_right_line)
        cv2.line(frame_ROI, (int(frame_ROI.shape[1] / 2) - 25, frame_ROI.shape[0]), (y_cv_vanishing_point, x_cv_theta), (232, 32, 1), 5)

        return y_cv_vanishing_point, x_cv_theta

    def only_one_line_detected(self, line_coefficients, frame_ROI, is_left_line=True):
        height_ROI = frame_ROI.shape[0]
        offset_center_road = 190    # experimental value
        x_cv_theta = -50  # the x_cv2 coordinate where we intersect -> wrt to ROI

        # transform in XoY coordinate
        y_theta = abs(x_cv_theta - height_ROI)
        x_theta = int((y_theta - line_coefficients[0]) / line_coefficients[1])

        y_cv_line = x_theta
        y_cv_vanishing_point = x_theta

        if is_left_line:
            y_cv_vanishing_point += offset_center_road
        else:
            y_cv_vanishing_point -= offset_center_road

        cv2.line(frame_ROI, (y_cv_vanishing_point, x_cv_theta), (y_cv_line, x_cv_theta), (200, 200, 200), 2)
        cv2.line(frame_ROI, (int(frame_ROI.shape[1] / 2) - 25, frame_ROI.shape[0]), (y_cv_vanishing_point, x_cv_theta),
                 (232, 32, 1), 5)

        return y_cv_vanishing_point, x_cv_theta

        pass

    def get_theta(self, frame_ROI_preprocessed, frame_ROI):
        left_lines, right_lines = self.get_and_filter_lines(frame_ROI_preprocessed, frame_ROI)
        found_line = False
        if left_lines and right_lines:
            # print("right and left")
            found_line = True
            left_line_coefficients = self.polyfit(left_lines, frame_ROI)
            right_line_coefficients = self.polyfit(right_lines, frame_ROI)
            y_cv_vanishing_point, x_cv_theta = self.both_lines_detected(left_line_coefficients, right_line_coefficients, frame_ROI)

        else:
            if right_lines:
                found_line = True
                right_line_coefficients = self.polyfit(right_lines, frame_ROI)
                y_cv_vanishing_point, x_cv_theta = self.only_one_line_detected(right_line_coefficients, frame_ROI, is_left_line=False)
            else:
                if left_lines:
                    found_line = True
                    left_line_coefficients = self.polyfit(left_lines, frame_ROI)
                    y_cv_vanishing_point, x_cv_theta = self.only_one_line_detected(left_line_coefficients, frame_ROI, is_left_line=True)

        if found_line:
            x_cv_center = frame_ROI.shape[0]
            y_cv_center = int(frame_ROI.shape[1] / 2) - 25  # camera lasata in partea stanga

            theta = math.degrees(math.atan((y_cv_center - y_cv_vanishing_point) / (x_cv_center - x_cv_theta)))
            return theta
        else:
            return -1000

    def drawLane(self, line, image, color_line):
        y1, x1, y2, x2 = line[0]
        radius = 10
        color_left_most_point = (0, 255, 0)
        color_right_most_point = (255, 0, 0)
        cv2.circle(image, (y1, x1), radius, color_left_most_point, 1)
        cv2.circle(image, (y2, x2), radius, color_right_most_point, 1)
        cv2.line(image, (y1, x1), (y2, x2), color_line, 2)

    def run(self):

        ret, frame = self.cap.read()
        theta_average = 0

        while True:
            start = time.time()

            # choosing our ROI
            # cv2.line(frame, (0, self.x_top - 5), (640, self.x_top - 5), (0, 0, 255), 2)
            frame_ROI = frame[self.x_top:, :]

            # preprocessing our ROI of the frame
            frame_ROI_preprocessed = self.preProcess(frame_ROI)
            theta = self.get_theta(frame_ROI_preprocessed, frame_ROI)
            if theta != -1000:  # we didn't detect any line
                theta_average = 0.6 * theta_average + 0.4 * theta
            print(theta_average)


            cv2.imshow("ROI", frame_ROI)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

            end = time.time()
            # print(end - start)
            ret, frame = self.cap.read()

LD = LaneDetection()

LD.run()