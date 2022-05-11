import cv2

cap = cv2.VideoCapture(0)

def preprocess(frame_ROI):
    gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    hist_eq = cv2.equalizeHist(gray)
    return hist_eq

while True:
    _, frame = cap.read()
    frame_ROI = frame[270:, :]
    frame_preprocessed = preprocess(frame_ROI)
    cv2.imshow("ROI Preprocess", frame_preprocessed)
    cv2.imshow("ROI", frame_ROI)
    cv2.waitKey(1)
