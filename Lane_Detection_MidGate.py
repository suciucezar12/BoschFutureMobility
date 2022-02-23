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

        frame_ROI_preprocessed = cv2.Canny(frame_ROI_blurred, 105, 255)

        return frame_ROI_preprocessed

    def hough_transform(self, frame_ROI_preprocessed, frame_ROI):
        left_side_ROI = frame_ROI_preprocessed[:, 0: int(frame_ROI_preprocessed.shape[1] / 2)]
        right_side_ROI = frame_ROI_preprocessed[:, int(frame_ROI_preprocessed.shape[1] / 2):]
        right_lines_detected = cv2.HoughLinesP(right_side_ROI, rho=1, theta=np.pi / 180, threshold=70, minLineLength=30,
                                         maxLineGap=70)
        left_lines_detected = cv2.HoughLinesP(left_side_ROI, rho=1, theta=np.pi / 180, threshold=70, minLineLength=30,
                                               maxLineGap=70)
        if right_lines_detected is not None:
            for line in right_lines_detected:
                line[0][0] += int(frame_ROI_preprocessed.shape[1] / 2)
                line[0][2] += int(frame_ROI_preprocessed.shape[1] / 2)

        if left_lines_detected is not None:
            for line_detected in left_lines_detected:
                self.drawLane(line_detected, frame_ROI, (255, 0, 0))

        if right_lines_detected is not None:
            for line_detected in right_lines_detected:
                self.drawLane(line_detected, frame_ROI, (0, 0, 255))

        return left_lines_detected, right_lines_detected

    def polyfit(self, lines):
        x_points = []
        y_points = []

        for line in lines:
            # x_points.append(line[0])
            # x_points.append(line[2])
            # y_points.append(line[1])
            # y_points.append(line[3])

            x_points.append(line[1])
            x_points.append(line[0])
            y_points.append(line[3])
            y_points.append(line[2])

        coeff = np.polynomial.polynomial.polyfit(x_points, y_points, 1)
        # print(type(coeff))
        return coeff

    def drawLane(self, line, image, color_line):
        x1, y1, x2, y2 = line[0]
        radius = 10
        color_left_most_point = (0, 255, 0)
        color_right_most_point = (255, 0, 0)
        cv2.circle(image, (x1, y1), radius, color_left_most_point, 1)
        cv2.circle(image, (x2, y2), radius, color_right_most_point, 1)
        cv2.line(image, (x1, y1), (x2, y2), color_line, 2)


    def run(self):

        ret, frame = self.cap.read()

        while True:
            start = time.time()
            cv2.line(frame, (0, self.x_top - 5), (640, self.x_top - 5), (0, 0, 255), 2)  # delimiting the ROI
            frame_ROI = frame[self.x_top:, :]

            frame_ROI_preprocessed = self.preProcess(frame_ROI)

            left_lines_detected, right_lines_detected = self.hough_transform(frame_ROI_preprocessed, frame_ROI)

            if left_lines_detected is not None:
                coeff = self.polyfit(left_lines_detected[0])
                if coeff is not None:
                    print("left_line: " + str(coeff[1]) + "*x + " + str(coeff[0]))
                    y1 = int(coeff[1] * 480 + coeff[0])
                    y2 = int(coeff[1] * 0 + coeff[0])
                    cv2.line(frame, (y1, x1), (y2, x2), (21, 32, 2), 5)
            if right_lines_detected is not None:
                coeff = self.polyfit(right_lines_detected[0])
                if coeff is not None:
                    print("right_line: " + str(coeff[1]) + "*x + " + str(coeff[0]))

            cv2.imshow("ROI", frame_ROI)
            # cv2.imshow("ROI preprocessed", frame_ROI_preprocessed)
            # cv2.imshow("Frame", frame)
            cv2.waitKey(1)
            end = time.time()
            print(end - start)
            ret, frame = self.cap.read()

LD = LaneDetection()

LD.run()