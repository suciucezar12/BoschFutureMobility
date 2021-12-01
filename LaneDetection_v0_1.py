import time
import cv2

time.sleep(0.1)
cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while ret:
    start = time.time()
    frame_copy = frame.copy()
    # take the bottom half of the frame
    frame_copy = frame[int(frame.shape[0] / 2):, :]
    cv2.imshow("Frame", frame_copy)

    # rgb -> grayscale
    frame_grayscale = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", frame_grayscale)

    # Apply Gaussian Blur
    frame_blurred = cv2.GaussianBlur(frame_grayscale, (3, 3), cv2.BORDER_DEFAULT)
    cv2.imshow("Gaussian Blur", frame_blurred)

    # Apply Canny
    frame_edge_detection = cv2.Canny(frame_blurred, 100, 200)
    cv2.imshow("Canny", frame_edge_detection)

    end = time.time()
    print(end - start)

    cv2.waitKey(1)
    ret, frame = cap.read()


