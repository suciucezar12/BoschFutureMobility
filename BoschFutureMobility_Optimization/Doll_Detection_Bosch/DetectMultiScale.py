import cv2
from skimage.feature import hog


class DetectMultiScale:

    def __init__(self, winSize=None, winStride=None, scale=None, nLevel=None):
        self.winSize = winSize
        self.winStride = winStride
        self.scale = scale
        self.nLevel = nLevel

    def sliding_window(self, image):

        result = []
        height, width = image.shape[0], image.shape[1]

        # while height >= self.winSize(0) and width >= self.winSize(1):
        #     result = [[(x, y, hog(image[y:y + self.winSize(0), x:x + self.winSize(1)], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True, visualize=False))
        #               for x in range(0, ((width - self.winSize(1)) % self.winStride(1) * self.winStride(1)), self.winStride(1))]
        #               for y in range(0, ((height - self.winSize(0)) % self.winStride(0) * self.winStride(0)), self.winStride(0))]
        for x in range(0, ((width - self.winSize(1)) % self.winStride(1) * self.winStride(1)), self.winStride(1)):
            for y in range(0, ((height - self.winSize(0)) % self.winStride(0) * self.winStride(0)), self.winStride(0)):
                cv2.rectangle(image, (x,y), (x + self.winSize(1), y + self.winSize(0)), (0, 255, 0), 2)
