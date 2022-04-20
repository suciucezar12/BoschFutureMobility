import numpy as np
import cv2


class Utils:

    def get_homography_matrix(self, src_points, dst_points, pixel_resolution):
        dst_points = np.array(
            [[int(y_cv / pixel_resolution), int(x_cv / pixel_resolution)] for [y_cv, x_cv] in dst_points],
            dtype=np.float32)
        return cv2.getPerspectiveTransform(src_points, dst_points)
