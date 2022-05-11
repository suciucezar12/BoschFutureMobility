import os
import random

import cv2
import numpy as np


class Classifier:

    def __init__(self, path_pos=None, path_neg=None,
                 winSize=None, blockSize=None, blockStride=None, cellSize=None, nbins=None):
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.hog = cv2.HOGDescriptor(_winSize=winSize, _blockSize=blockSize, _blockStride=blockStride,
                                     _cellSize=cellSize, _nbins=nbins)
        self.svm = cv2.ml.SVM_create()
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setC(2.67)
        self.svm.setGamma(5.383)

    def test(self):
        pass

    # train with all samples: pos and neg shuffled
    def train(self):
        samples = []
        labels = []
        print("TRAINING")
        if self.path_pos is not None:
            for image in os.listdir(self.path_pos):
                if image.endswith(".jpg"):
                    image = cv2.imread(self.path_pos + "/" + image)
                    image = cv2.resize(image, (64,64))
                    # cv2.imshow("sample pos", image)
                    # cv2.waitKey(0)
                    samples.append(np.ravel(self.hog.compute(image)).tolist())
                    labels.append(1)


            for image in os.listdir(self.path_neg):
                if image.endswith(".jpg"):
                    image = cv2.imread(self.path_neg + "/" + image)
                    # cv2.imshow("sample neg", image)
                    # cv2.waitKey(0)
                    image = cv2.resize(image, (64, 64))
                    samples.append(np.ravel(self.hog.compute(image)).tolist())
                    labels.append(0)

        # TO DO: apply a shuffle
        l = list(zip(samples, labels))
        random.shuffle(l)
        samples, labels = zip(*l)
        print(len(samples))
        print(len(labels))

        self.svm.train(np.matrix(samples, dtype=np.float32), cv2.ml.ROW_SAMPLE, np.array(labels))
        self.svm.save('./svm.xml')

cls = Classifier(path_pos="./train_pos_set", path_neg="./train_neg_set",
                winSize=(64, 64), blockSize=(16, 16), blockStride=(8, 8), cellSize=(8, 8), nbins=9)
# cls.train()
