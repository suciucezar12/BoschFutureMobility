import numpy as np
import cv2


class Line:

    def __init__(self, coords, coeff):
        self.coords = coords
        self.coeff = coeff

class Utils:

    def get_homography_matrix(self, src_points, dst_points, pixel_resolution):
        dst_points = np.array(
            [[int(y_cv / pixel_resolution), int(x_cv / pixel_resolution)] for [y_cv, x_cv] in dst_points],
            dtype=np.float32)
        return cv2.getPerspectiveTransform(src_points, dst_points)

    def filter_lines(self, lines_candidate, frame_ROI, frame_ROI_IPM):
        left_lines = []
        right_lines = []
        horizontal_lines = []
        width_ROI = 640
        height_ROI = 160
        # used for intercept_oX criteria
        y_cv_margin = 80  # offset wrt to the center vertical line
        margin_y_cv_left = int(width_ROI / 2) - y_cv_margin
        margin_y_cv_right = int(width_ROI / 2) + y_cv_margin

        for line in lines_candidate:
            y1_cv, x1_cv, y2_cv, x2_cv = line[0]
            # centroid = [(y1_cv + y2_cv) // 2, (x1_cv + x2_cv) // 2]
            if y1_cv != y2_cv:
                coeff = np.polynomial.polynomial.polyfit((y1_cv, y2_cv), (x1_cv, x2_cv), deg=1)
                # ---------------------------------
                # coeff = []
                # try:
                #     slope = (x2_cv - x1_cv) / (y2_cv - y1_cv)
                # except OverflowError:
                #     slope = 10000
                # coeff.append(y1_cv - slope * x1_cv)
                # coeff.append(slope)
                # print(coeff)
                # ---------------------------------
                if coeff is not None:
                    # coeff[1] -> slope in XoY coordinates
                    # coeff[0] -> intercept_oY in XoY coordinates
                    if coeff[1] != 10000:
                        if abs(coeff[1]) >= 0.7:  # slope = +-0.2 -> +-11.3 degrees
                            # OverFlowError when we get horizontal lines
                            try:
                                # intercept_oX = - int(coeff[0] / coeff[1])
                                # print((self.height_ROI - coeff[0]) / coeff[1])
                                intercept_oX = int((height_ROI - coeff[0]) / coeff[1])
                            except OverflowError:
                                intercept_oX = 30000  # some big value
                            # print("y = {}*x + {}".format(coeff[1], coeff[0]))
                            # print(intercept_oX)
                            if 0 <= intercept_oX <= margin_y_cv_left:  # left line
                                left_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                # self.left_lines.append(line)
                                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 0, 0), 2)

                            if margin_y_cv_right <= intercept_oX <= width_ROI:  # right line
                                right_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                # self.right_lines.append(line)
                                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 0, 255), 2)

                            # check by theta and intercept_oX (last criteria)
                            if coeff[1] <= -0.2:  # candidate left line
                                if 0 <= intercept_oX <= margin_y_cv_right:
                                    left_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                    # self.left_lines.append(line)
                                    cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (255, 0, 0), 2)

                            if coeff[1] >= 0.2:  # candidate right line
                                if margin_y_cv_left <= intercept_oX <= width_ROI:
                                    right_lines.append(Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff))
                                    # self.right_lines.append(line)
                                    cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 0, 255), 2)
                        else:
                            if abs(coeff[1]) <= 0.3:
                                horizontal_lines.append(line)
                                # self.horizontal_lines.append(line)

        return left_lines, right_lines, horizontal_lines

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
                # cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)

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

    def translation_IPM(self, line_IPM, width_road, left_lane=None):
        if left_lane:
            offset = width_road
        else:
            offset = - width_road
        y1_cv, x1_cv, y2_cv, x2_cv = line_IPM
        return [y1_cv + offset, x1_cv, y2_cv + offset, x2_cv]
