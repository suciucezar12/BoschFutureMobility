import time
import cv2


class LaneDetection:

    def __init__(self):
        time.sleep(0.1)
        self.cap = cv2.VideoCapture(0)

    # def draw_ROI(self, frame):
    #     width_frame = 640
    #     height_frame = 480
    #
    #     # select the 4 points of the trapezoid
    #     pt1 = (0, 640)
    #     pt2 = (50, 600)
    #     pt3 = (590, 600)
    #     pt4 = (480, 640)
    #
    #     cv2.line(frame, pt1, pt2, (255, 255, 0), 3)
    #     cv2.line(frame, pt2, pt3, (255, 255, 0), 3)
    #     cv2.line(frame, pt3, pt4, (255, 255, 0), 3)
    #     cv2.line(frame, pt4, pt1, (255, 255, 0), 3)
    #
    #     return frame

    def run(self):

        ret, frame = self.cap.read()

        while True:

            print(str(frame.shape[0]) + " " + str(frame.shape[1]))
            # Selecting ROI -> looking for a trapezoid where our lanes would always appear
            # base of the trapezoid is actually the bottom line of our frame
            # frame = self.draw_ROI(frame)

            pt1 = (0, 640)
            pt2 = (50, 600)
            pt3 = (590, 600)
            pt4 = (480, 640)

            frame_copy = frame.copy()

            cv2.line(frame_copy, (0, 400), (50, 500), (255, 255, 0), 3)
            # cv2.line(frame_copy, (50, 600), (590, 600), (255, 255, 0), 3)
            # cv2.line(frame_copy, (590, 600), (480, 639), (255, 255, 0), 3)
            # cv2.line(frame_copy, (480, 639), (0, 100), (255, 255, 0), 3)

            cv2.imshow("Frame", frame_copy)
            cv2.waitKey(1)

            ret, frame = self.cap.read()


LD = LaneDetection()

LD.run()
