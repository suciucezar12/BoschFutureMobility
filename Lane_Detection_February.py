import time
import cv2


class LaneDetection:

    def __init__(self):

        time.sleep(0.1)
        self.cap = cv2.VideoCapture(0)

        self.y_top_left_trapezoid = 90
        self.x_top_trapezoid = 300
        self.y_top_right_trapezoid = 550

    def drawROI(self, frame):
        cv2.line(frame, (0, 480), (self.y_top_left_trapezoid, self.x_top_trapezoid), (0, 255, 0), 2)
        cv2.line(frame, (self.y_top_left_trapezoid, self.x_top_trapezoid), (self.y_top_right_trapezoid, self.x_top_trapezoid), (0, 255, 0), 2)
        cv2.line(frame, (self.y_top_right_trapezoid, self.x_top_trapezoid), (640, 480), (0, 255, 0), 2)


    def run(self):

        ret, frame = self.cap.read()

        while True:

            print(str(frame.shape[0]) + " " + str(frame.shape[1]))
            # Selecting ROI -> looking for a trapezoid where our lanes would always appear
            # base of the trapezoid is actually the bottom line of our frame
            self.drawROI(frame)

            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

            ret, frame = self.cap.read()


LD = LaneDetection()

LD.run()
