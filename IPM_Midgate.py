import time

import cv2
import numpy as np

x_cv_ROI = 270
cap = cv2.VideoCapture(0)

pixel_resolution = 0.122  # centimeters per pixel
height_ROI_IPM = 210    # calculated related to pixel_resolution and the real dimensions
width_ROI_IPM = 547
width_ROI = 640
height_ROI = 210


ret, frame = cap.read()

# clockwise selection of corners
src_points = np.array([[0, 0], [width_ROI, 0], [width_ROI, height_ROI], [0, height_ROI]], dtype=np.float32)
dst_points = np.array([[0, 3.1], [66.7, 0], [53.5, 24.6], [11, 25.6]])
dst_points = np.array([[int(y_cv / pixel_resolution), int(x_cv / pixel_resolution)] for [y_cv, x_cv] in dst_points], dtype=np.float32)
H = cv2.getPerspectiveTransform(src_points, dst_points)

while True:

    start = time.time()

    cv2.line(frame, (0, x_cv_ROI + 5), (640, x_cv_ROI + 5), (0, 0, 255), 1)
    frame_ROI = frame[x_cv_ROI:, :]
    frame_ROI_IPM = cv2.warpPerspective(frame_ROI, H, (width_ROI_IPM, height_ROI_IPM), flags=cv2.INTER_NEAREST)  # CV2.INTER_LINEAR

    # test transformation on a set of points
    set_points = np.array([[[20, 30]]], dtype=np.float32)
    print(set_points[0][0])
    # cv2.circle(frame_ROI, tuple(set_points[0][0]), 5, (0, 255, 0), 1)
    # transform set points in new plane
    IPM_set_points = cv2.perspectiveTransform(set_points, H)[0]
    # cv2.circle(frame_ROI_IPM, tuple(IPM_set_points[0][0]), 5, (255, 0, 0), 1)

    cv2.imshow("IPM", frame_ROI_IPM)
    cv2.imshow("ROI", frame_ROI)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    print(time.time() - start)

    ret, frame = cap.read()