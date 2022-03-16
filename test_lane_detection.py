import cv2
import numpy as np

class LDTest:

    def __init__(self):
        self.x_cv_ROI = 270
        self.cap = cv2.VideoCapture(0)

    def draw_line(self, line, color, image):
        y1_cv, x1_cv, y2_cv, x2_cv = line[0]
        radius = 5
        color_left_most_point = (0, 255, 0)  # GREEN for left_most point
        color_right_most_point = (255, 0, 0)  # BLUE fpr right_most point
        cv2.circle(image, (y1_cv, x1_cv), radius, color_left_most_point, 1)
        cv2.circle(image, (y2_cv, x2_cv), radius, color_right_most_point, 1)
        cv2.line(image, (y1_cv, x1_cv), (y2_cv, x2_cv), color, 2)

    def preprocess(self, frame_ROI):
        gray = cv2.cvtColor(frame_ROI, code=cv2.COLOR_BGR2GRAY)
        contrast = cv2.convertScaleAbs(gray, alpha=1.3, beta=0)

        # cv2.imshow("ROI preprocessed", gray)
        # cv2.imshow("Contrast", contrast)

        canny_gray = cv2.Canny(gray, 30, 255)
        canny_contrast = cv2.Canny(contrast, 30, 255)

        return canny_contrast, canny_gray

    def show_lines(self, lines, image):
        if lines is not None:
            for line in lines:
                self.draw_line(line, (255, 0, 0), image)

    def detect_lines(self, canny_contrast, canny_gray):
        lines_contrast = cv2.HoughLinesP(canny_contrast, rho=1, theta=np.pi / 180, threshold=50, minLineLength=35,
                        maxLineGap=80)
        lines_gray = cv2.HoughLinesP(canny_gray, rho=1, theta=np.pi / 180, threshold=40, minLineLength=35,
                        maxLineGap=80)
        self.show_lines(lines_contrast, canny_contrast)
        self.show_lines(lines_gray, canny_gray)

    def run(self):

        ret, frame = self.cap.read()
        while True:
            frame_ROI = frame[self.x_cv_ROI:, :]
            canny_contrast, canny_gray = self.preprocess(frame_ROI)
            self.detect_lines(canny_contrast, canny_gray)
            cv2.imshow("ROI preprocessed", canny_gray)
            cv2.imshow("Contrast", canny_contrast)
            cv2.imshow("ROI", frame_ROI)
            cv2.waitKey(1)
            _, frame = self.cap.read()

testld = LDTest()
testld.run()