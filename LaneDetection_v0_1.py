import time
import cv2

time.sleep(0.1)
cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while ret:
    frame_copy = frame.copy()
    cv2.imshow("Frame", frame_copy)

    # print("H = " + str(frame_copy.shape[0]) + ", W = " +  str(frame_copy.shape[1]))
    # resize the image
    #

    # rgb -> grayscale
    frame_grayscale = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", frame_grayscale)

    # Apply Gaussian Blur
    frame_blurred = cv2.GaussianBlur(frame_grayscale, (5, 5), cv2.BORDER_DEFAULT)
    cv2.imshow("Gaussian Blur", frame_blurred)

    # Apply Sobel on X axis
    frame_edge_detection = cv2.Sobel(frame_blurred, cv2.CV_64F, 1, 0, ksize=5)
    cv2.imshow("Sobel", frame_edge_detection)
    cv2.waitKey(1)


