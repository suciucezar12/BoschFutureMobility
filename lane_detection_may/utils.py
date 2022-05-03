import numpy as np
import cv2
import math

class Line:

    def __init__(self, coords, coeff):
        self.coords = coords
        self.coeff = coeff

class Utils:

    def draw_line(self, line, color, image):
        y1_cv, x1_cv, y2_cv, x2_cv = line[0]
        radius = 5
        color_left_most_point = (0, 255, 0)  # GREEN for left_most point
        color_right_most_point = (255, 0, 0)  # BLUE fpr right_most point
        cv2.circle(image, (y1_cv, x1_cv), radius, color_left_most_point, 1)
        cv2.circle(image, (y2_cv, x2_cv), radius, color_right_most_point, 1)
        cv2.line(image, (y1_cv, x1_cv), (y2_cv, x2_cv), color, 1)

    def get_homography_matrix(self, src_points, dst_points, pixel_resolution):
        dst_points = np.array(
            [[int(y_cv / pixel_resolution), int(x_cv / pixel_resolution)] for [y_cv, x_cv] in dst_points],
            dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        return H

    def preprocessing(self, frame_ROI):
        frame_ROI_gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
        # frame_ROI_blurred = cv2.GaussianBlur(frame_ROI_gray, (5, 5), 0)
        gaussian = cv2.GaussianBlur(frame_ROI_gray, (5,5), sigmaX=0)
        contrast = cv2.convertScaleAbs(gaussian, alpha=1.6)
        frame_ROI_canny = cv2.Canny(contrast, 180, 255)
        return frame_ROI_canny

    def estimate_lane(self, lines, height_ROI, frame_ROI, frame_ROI_IPM=None):
        if len(lines) > 0:
            # lines are type of class Line
            x_points = []
            y_points = []
            for line in lines:
                y1_cv, x1_cv, y2_cv, x2_cv = line.coords
                x_points.append(y1_cv)
                x_points.append(y2_cv)
                y_points.append(x1_cv)
                y_points.append(x2_cv)

                # create more data points on the line for better precision in our line estimation
                num = 5
                for y_cv in np.linspace(y1_cv, y2_cv, num):
                    x_points.append(y_cv)
                    x_cv = line.coeff[1] * y_cv + line.coeff[0]
                    y_points.append(x_cv)
                    # cv2.circle(frame_ROI, (int(y_cv), int(x_cv)), 5, (0, 0, 255), 2)
                # -------------------------------------------------------------------------------

            # estimate our lane
            coeff = np.polynomial.polynomial.polyfit(y_points, x_points, deg=1)
            # print("x = {}*y + {}".format(coeff[1], coeff[0]))

            if coeff is not None:
                # get our coordinates expanded from top to the bottom
                x1_cv = height_ROI
                x2_cv = 0
                y1_cv = int(coeff[1] * x1_cv + coeff[0])
                y2_cv = int(coeff[1] * x2_cv + coeff[0])
                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)

                return [y1_cv, x1_cv, y2_cv, x2_cv]
            else:
                return None
        else:
            return None

    def get_line_IPM(self, line, H):
        y1_cv, x1_cv, y2_cv, x2_cv = line
        src_points = np.array([[[y1_cv, x1_cv], [y2_cv, x2_cv]]], dtype=np.float32)
        dest_points = cv2.perspectiveTransform(src_points, H)[0]
        return [int(dest_points[0][0]), int(dest_points[0][1]), int(dest_points[1][0]), int(dest_points[1][1])]

    def get_line_eq(self, p1, p2):
        y1, x1 = p1
        y2, x2 = p2
        if y1 != y2:
            slope = float((x2 - x1) / (y2 - y1))
        else:
            slope = np.sign(y2 - y1) * 3000000
        intercept = float(y1 - slope * x1)
        return [intercept, slope]

    def line_intersection(self, eq1, eq2):
        b1, m1 = eq1
        b2, m2 = eq2
        y = (b1 - b2) / (m2 - m1)
        x = m1 * y + b1
        return int(y), int(x)

    def translation_IPM(self, line_IPM, width_road, left_lane=None):
        if left_lane:
            offset = width_road
        else:
            offset = - width_road
        y1_cv, x1_cv, y2_cv, x2_cv = line_IPM
        return [y1_cv + offset, x1_cv, y2_cv + offset, x2_cv]
