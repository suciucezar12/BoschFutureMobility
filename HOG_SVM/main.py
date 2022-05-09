import cv2

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

while True:

    width_box, height_box = 200, 200
    y = frame.shape[0] // 2
    x = frame.shape[1] // 2

    # draw box
    x1, y1 = x - width_box // 2, y - height_box // 2
    x2, y2 = x + width_box // 2, y + height_box // 2

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    _, frame = cap.read()


