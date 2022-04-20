import numpy as np
import cv2


class Line:

    def __init__(self, line):
        y1_cv, x1_cv, y2_cv, x2_cv = line[0]
        self.coords = [y1_cv, x1_cv, y2_cv, x2_cv]
        # get slope --------------------------------------------------
        # check if vertical line
        try:
            self.slope = float((x2_cv - x1_cv) / (y2_cv - y1_cv))
        except OverflowError:
            if x2_cv > x1_cv:
                self.slope = 30000  # some big value:  atan(slope) -> infinity
            else:
                self.slope = -30000
        # get intercept_oY --------------------------------------------
        self.intercept_oY = y1_cv - self.slope * x1_cv
        # -------------------------------------------------------------
        self.centroid = [(y1_cv + y2_cv) // 2, (x1_cv + x2_cv) // 2]

class Utils:

    def get_homography_matrix(self, src_points, dst_points, pixel_resolution):
        dst_points = np.array(
            [[int(y_cv / pixel_resolution), int(x_cv / pixel_resolution)] for [y_cv, x_cv] in dst_points],
            dtype=np.float32)
        return cv2.getPerspectiveTransform(src_points, dst_points)
