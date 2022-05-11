import os
import random
import numpy as np

import cv2

class HogDescriptor:

    def __init__(self, winSize=None, blockSize=None, blockStride=None, cellSize=None, nbins=None,
                 c=None, gamma=None):
        self.hog = cv2.HOGDescriptor(_winSize=winSize, _blockSize=blockSize, _blockStride=blockStride,
                                     _cellSize=cellSize, _nbins=nbins)
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setC(c)
        self.svm.setGamma(gamma)

    def train(self, path_pos=None, path_neg=None):
        fd = []
        label = []  # 1 -> positive, 0 -> negative
        if path_pos is not None:
            for image in os.listdir(path_pos):
                if image.endswith(".jpg"):
                    image = cv2.imread(path_pos + "/" + image)
                    fd.append(self.hog.compute(image))
                    label.append(1)
        if path_neg is not None:
            for image in os.listdir(path_neg):
                if image.endswith(".jpg"):
                    image = cv2.imread(path_neg + "/" + image)
                    fd.append(self.hog.compute(image))
                    label.append(0)

        # l = list(zip(fd, label))
        # random.shuffle(l)
        # fd, label = zip(*l)
        # label = list(label)
        # fd = np.matrix(fd, dtype=np.float32)
        label = np.array(label)
        print(len(np.array(fd)[0]))
        print(label)
        self.svm.train(np.array(fd), cv2.ml.ROW_SAMPLE, label)
        self.svm.save("./svm.dat")

    def test(self, path_test=None):
        pass

    def convert_fd(self, fd):
        fd_conv = []
        for block in fd:
            block_conv = []



HD = HogDescriptor(winSize=(64,64), blockSize=(16,16), blockStride=(8,8), cellSize=(8,8), nbins=9,
                   c=2.57, gamma=2)
HD.train(path_pos="./train_data_set")


# help(cv2.HOGDescriptor)
#
# cap = cv2.VideoCapture(0)
#
# hog = cv2.HOGDescriptor(_winSize=(64, 128), _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=9)
# ret, frame = cap.read()
#
# while True:
#
#     frame = cv2.resize(frame, (64, 128))
#     hist = hog.compute(frame) # feature descriptor
#     cv2.imshow("Frame", frame)
#     cv2.waitKey(0)
#     _, frame = cap.read()

