import cv2
import numpy as np
import time

class LaneDetection:

    def __init__(self):
        ''' Matrix used for IPM '''
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
        cv2.imshow("ROI_Blurred", frame_ROI_blurred)

        frame_ROI_preprocessed = cv2.Canny(frame_ROI_blurred, 50, 200)

        return frame_ROI_preprocessed

    def hough_transform(self, frame_ROI_preprocessed):
        lines_detected = cv2.HoughLinesP(frame_ROI_preprocessed, rho=1, theta=np.pi / 180, threshold=100, minLineLength=30,
                                         maxLineGap=50)
        return lines_detected

    def drawLane(self, line, image):
        x1, y1, x2, y2 = line[0]
        radius = 10
        color_left_most_point = (0, 255, 0)
        color_right_most_point = (255, 0, 0)
        cv2.circle(image, (x1, y1), radius, color_left_most_point, 1)
        cv2.circle(image, (x2, y2), radius, color_right_most_point, 1)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


    def run(self):

        ret, frame = self.cap.read()
        while True:

            cv2.line(frame, (0, self.x_top - 5), (640, self.x_top - 5), (0, 0, 255), 2)  # delimiting the ROI
            frame_ROI = frame[self.x_top:, :]

            frame_ROI_preprocessed = self.preProcess(frame_ROI)

            lines_detected = self.hough_transform(frame_ROI_preprocessed)
            if lines_detected is not None:
                for line_detected in lines_detected:
                    self.drawLane(line_detected, frame_ROI)

            cv2.imshow("ROI", frame_ROI)
            cv2.imshow("ROI preprocessed", frame_ROI_preprocessed)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            ret, frame = self.cap.read()

LD = LaneDetection()

LD.run()