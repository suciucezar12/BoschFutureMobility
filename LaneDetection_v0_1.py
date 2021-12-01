import time
import cv2

time.sleep(0.1)
cap = cv2.VideoCapture(0)



while ret:
    ret, frame = cap.read()
    start = time.time()
    frame_copy = frame.copy()
    frame_copy = frame[int(frame.shape[0] / 2):, :]
    cv2.imshow("Frame", frame_copy)

    # print("H = " + str(frame_copy.shape[0]) + ", W = " +  str(frame_copy.shape[1]))
    # resize the image
    #

    # rgb -> grayscale
    frame_grayscale = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", frame_grayscale)

    # Apply Gaussian Blur
    frame_blurred = cv2.GaussianBlur(frame_grayscale, (7, 7), cv2.BORDER_DEFAULT)
    cv2.imshow("Gaussian Blur", frame_blurred)

    # Apply Canny
    # gX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    frame_edge_detection = cv2.Canny(frame_blurred, 100, 200)
    # frame_edge_detection = cv2.Sobel(frame_blurred, cv2.CV_32F, 1, 0, ksize=3)
    cv2.imshow("Canny", frame_edge_detection)
    end = time.time()
    print(end - start)
    cv2.waitKey(1)


