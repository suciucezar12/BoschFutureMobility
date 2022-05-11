import os
import random

import numpy as np
import cv2
from sklearn import svm
from skimage.feature import hog

class DollDetection:

    def __init__(self):
        self.clf = svm.SVC()
        pass

    def train(self, path_pos=None, path_neg=None):
        samples = []
        labels = []

        # positive samples
        if path_pos is not None:
            for image in os.listdir(path_pos):
                if image.endswith(".jpg"):
                    image = cv2.imread(path_pos + "/" + image)
                    fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm='L2', feature_vector=True, visualize=False)
                    samples.append(fd)
                    labels.append(1)

        # negative samples
        if path_neg is not None:
            for image in os.listdir(path_neg):
                if image.endswith(".jpg"):
                    image = cv2.imread(path_neg + "/" + image)
                    fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2), block_norm='L2', feature_vector=True, visualize=False)
                    samples.append(fd)
                    labels.append(0)

        # shuffle data
        l = list(zip(samples, labels))
        random.shuffle(l)
        samples, labels = zip(*l)

        self.clf.fit(samples, labels)


DD = DollDetection()
DD.train(path_neg="./train_neg_set", path_pos="./train_pos_set")
