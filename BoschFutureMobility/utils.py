import numpy as np
import cv2


def get_homography_matrix(src_points, dst_points, pixel_resolution):
    """
    :return: Homography matrix use for IPM
    """
    dst_points = np.array(
        [[int(y_cv / pixel_resolution), int(x_cv / pixel_resolution)] for [y_cv, x_cv] in dst_points],
        dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    return H