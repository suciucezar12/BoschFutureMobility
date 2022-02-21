import time
import numpy as np
import cv2

class LaneDetection:

    def __init__(self):
        ''' Matrix used for IPM '''
        self.x_top = 270    # Coordinates of the polygon we use for creating the Homography matrix
        self.y_left_top = 80
        self.y_right_top = 560
        self.input_coordinates_IPM = np.array([[0, 480], [self.y_left_top, self.x_top], [self.y_right_top, self.x_top], [640, 480]], dtype=np.float32)
        self.output_coordinates_IPM = np.array([[199, 36], [417, 0], [439, 444], [205, 410]], dtype=np.float32) # Output coordinates calculated manually in our flat real word plane of the road
        self.matrix_IPM = cv2.getPerspectiveTransform(self.input_coordinates_IPM, self.output_coordinates_IPM)
        ''' ================================================================================================================================ '''

        time.sleep(1)
        self.cap = cv2.VideoCapture(0)

    def get_IPM_frame(self, frame):
        frame_IPM_width = 450
        frame_IPM_height = 450
        frame_IPM = cv2.warpPerspective(frame, self.matrix_IPM, (frame_IPM_width, frame_IPM_height), flags=cv2.INTER_LINEAR)
        rotation_matrix = cv2.getRotationMatrix2D((frame_IPM_width / 2, frame_IPM_height / 2), 90, 1.0)
        frame_IPM_rotated = cv2.warpAffine(frame_IPM, rotation_matrix, (frame_IPM_width, frame_IPM_height))
        margin_x_crop = 28
        height_crop = 245
        frame_IPM_final = frame_IPM_rotated[: height_crop, margin_x_crop: frame_IPM_rotated.shape[0] - margin_x_crop]

        return frame_IPM_final

    def preProcess(self, frame_IPM):
        frame_gray = cv2.cvtColor(frame_IPM, cv2.COLOR_BGR2GRAY)
        frame_blurred = cv2.GaussianBlur(frame_gray, (11, 11), 0)
        frame_edge = cv2.Canny(frame_blurred, 50, 200)
        return frame_edge

    def drawLine(self, image, line, color, margin_y):
        x1, y1, x2, y2 = line[0]
        x1 += margin_y
        x2 += margin_y
        cv2.line(image, (x1, y1), (x2, y2), color, 3)

    def get_candidate_lines(self, frame_preprocessed):

        # divide in half on vertical axis our image
        # left side is for detecting left lines
        left_side_frame = frame_preprocessed[:, : int(frame_preprocessed.shape[1] / 2)]
        # right side is for detecting right lines
        right_side_frame = frame_preprocessed[:, int(frame_preprocessed.shape[1] / 2):]

        cv2.imshow("Left_side", left_side_frame)
        cv2.imshow("Right_side", right_side_frame)



        left_lines_candidate = cv2.HoughLinesP(left_side_frame, rho=1, theta=np.pi / 180, threshold=35, minLineLength=10,
                                          maxLineGap=15)
        right_lines_candidate = cv2.HoughLinesP(right_side_frame, rho=1, theta=np.pi / 180, threshold=35, minLineLength=10,
                                               maxLineGap=15)

        return left_lines_candidate, right_lines_candidate

    def run(self):

        ret, frame = self.cap.read()

        while True:

            start = time.time()  # measure (time_computing / frame)

            # cv2.polylines(frame, np.int32([self.input_coordinates_IPM]), True, (0, 255, 255))

            # frame after applying IPM and cropping
            frame_IPM = self.get_IPM_frame(frame)
            # draw the vertical axis on the center of the IPM_frame
            x_vertical_axis = int(frame_IPM.shape[1] / 2)
            cv2.line(frame_IPM, (x_vertical_axis, 480), (x_vertical_axis, 0), (255, 0, 255), 5)
            # cv2.line()
            # frame after applying preprocessing
            frame_preprocessed = self.preProcess(frame_IPM)
            # choose candidate lines
            left_lines_candidate, right_lines_candidate = self.get_candidate_lines(frame_preprocessed)

            if left_lines_candidate is not None:
                for line in left_lines_candidate:
                    self.drawLine(frame_IPM, line, (0, 0, 255), 0)

            if right_lines_candidate is not None:
                for line in right_lines_candidate:
                    x1, y1, x2, y2 = line[0]
                    self.drawLine(frame_IPM, line, (255, 0, 0), int(frame_IPM.shape[1] / 2))


            cv2.imshow("IPM", frame_IPM)
            cv2.imshow("IPM Preprocessed", frame_preprocessed)
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

            end = time.time()
            print(end - start)

            ret, frame = self.cap.read()



LD = LaneDetection()
LD.run()
