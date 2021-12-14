import time
import cv2


class LaneDetection:

    def __init__(self):
        time.sleep(0.1)
        self.cap = cv2.VideoCapture(0)

    def apply_filter(self, ret, frame):
        start = time.time()
        frame_copy = frame.copy()

        frame_copy = frame[int(frame.shape[0] / 2):, :]
        cv2.imshow("Frame", frame_copy)

        # rgb -> grayscale
        frame_grayscale = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale", frame_grayscale)

        # Apply Gaussian Blur
        frame_blurred = cv2.GaussianBlur(frame_grayscale, (7, 7), cv2.BORDER_DEFAULT)
        cv2.imshow("Gaussian Blur", frame_blurred)

        # Apply Canny
        frame_edge_detection = cv2.Canny(frame_blurred, 100, 200)
        cv2.imshow("Canny", frame_edge_detection)

        end = time.time()
        print(end - start)

        cv2.waitKey(1)
        return frame_edge_detection

    def run(self):
        ret, frame = self.cap.read()
        while ret:
            self.apply_filter(ret, frame)
            ret, frame = self.cap.read()


LD = LaneDetection()
LD.run()
