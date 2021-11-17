from time import sleep
from picamera import PiCamera

# Capture to a file
# camera = PiCamera()
# camera.resolution = (1024, 768)
# camera.start_preview()
#
# sleep(2) # camera warm-up time
# camera.capture('foo.jpg')




# Capture to a stream
from io import BytesIO
# # create an in-memory stream
# my_stream = BytesIO()
# camera = PiCamera()
# camera.start_preview()
# sleep(2)
# camera.capture(my_stream, 'jpeg')




# stream.2
my_file = open('my_image.jpg', 'wb')
camera = PiCamera()
camera.start_preview()
sleep(2)
camera.capture(my_file) # my_flie.flush() is called
my_file.close()





