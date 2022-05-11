import cv2


class DetectMultiScale:

    def __init__(self, winSize=None, winStride=None, scale=None, nLevel=None):
        self.winSize = winSize
        self.winStride = winStride
        self.scale = scale
        self.nLevel = nLevel

    def run(self, image):
        height, width = image.shape