import cv2

cap = cv2.VideoCapture(0)

def preprocess(frame_ROI):
    gray = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
    

while True:
    _, frame = cap.read()
    frame_ROI = frame[270:, :]
    cv2.imshow("ROI", frame_ROI)
    cv2.waitKey(1)
