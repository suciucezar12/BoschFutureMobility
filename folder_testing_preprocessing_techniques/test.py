import time
import cv2

cap = cv2.VideoCapture(0)

def preprocess(frame_ROI):
    gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    cv2.imshow("canny + gray", cv2.Canny(gray, 100, 250))
    hist_eq = cv2.equalizeHist(gray)
    gaussian = cv2.GaussianBlur(hist_eq, ksize=(3,3), sigmaX=0)
    canny = cv2.Canny(gaussian, 100, 250)
    cv2.imshow("canny + hist eq", canny)
    return gaussian


def preprocess1(frame_ROI):
    gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    # cv2.imshow("canny + gray", cv2.Canny(gray, 100, 250))
    hist_eq = cv2.equalizeHist(gray)
    gaussian = cv2.GaussianBlur(hist_eq, ksize=(3,3), sigmaX=0)
    _, otsu = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu

while True:
    _, frame = cap.read()
    start = time.time()
    frame_ROI = frame[270:, :]
    frame_preprocessed = preprocess(frame_ROI)
    cv2.imshow("ROI Preprocess", frame_preprocessed)
    # cv2.imshow("ROI", frame_ROI)
    end = time.time()
    print("time = {}".format(end - start))
    cv2.waitKey(1)
