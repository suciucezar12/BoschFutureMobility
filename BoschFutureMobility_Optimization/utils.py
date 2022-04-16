import numpy as np
import cv2
import math


class Line:

    def __init__(self, coords, coeff):
        self.coords = coords
        self.coeff = coeff

class Utils:

    def get_homography_matrix(self, src_points, dst_points, pixel_resolution):
        dst_points = np.array(
            [[int(y_cv / pixel_resolution), int(x_cv / pixel_resolution)] for [y_cv, x_cv] in dst_points],
            dtype=np.float32)
        H = cv2.getPerspectiveTransform(src_points, dst_points)
        return H

    def preprocessing(self, frame_ROI):
        frame_ROI_gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
        frame_ROI_blurred = cv2.GaussianBlur(frame_ROI_gray, (5, 5), 0)
        frame_ROI_canny = cv2.Canny(frame_ROI_blurred, 180, 255)
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
                    cv2.circle(frame_ROI, (int(y_cv), int(x_cv)), 3, (0, 0, 255), 1)
                # -------------------------------------------------------------------------------

            # estimate our lane
            coeff = np.polynomial.polynomial.polyfit(y_points, x_points, deg=1)
            print("y = {}*x + {}".format(coeff[1], coeff[0]))

            if coeff is not None:
                # get our coordinates expanded from top to the bottom
                x1_cv = 0
                x2_cv = height_ROI
                y1_cv = int((x1_cv - coeff[0]) / coeff[1])
                y2_cv = int((x2_cv - coeff[0]) / coeff[1])
                cv2.line(frame_ROI, (y1_cv, x1_cv), (y2_cv, x2_cv), (0, 255, 0), 3)

                return Line((y1_cv, x1_cv, y2_cv, x2_cv), coeff)
            else:
                return None
        else:
            return None
