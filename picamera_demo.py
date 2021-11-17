from time import sleep
from picamera import PiCamera

camera = Camera()
camera.resolution = (1024, 768)
camera.start_preview()

sleep(2) # camera warm-up time
camera.capture('foo.jpg')
