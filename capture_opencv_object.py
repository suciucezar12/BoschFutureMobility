import io
import time
import picamera
import cv2
import numpy as np

# Create the in-memory stream
stream = io.BytesIO()
with picamera.PiCamera() as camera:
    camera.start_preview()
    time.sleep(2)
    camera.capture(stream, format='jpeg')

# Construct a numpy array from the stream
data = np.fromstring(stream.getvalue(), dtype='uint8')
# "Decode" the image from the array. preserving colour
image = cv2.imdecode(data, 1)
# OpenCV returns an array with data in BGR order
cv2.imshow("Image", image)
cv2.waitKey(0)

