import pickle
import re

import cv2
import numpy as np
from Classifier import Classifier
import xml.etree.ElementTree as ET


class DetectMultiScale:

    def __init__(self, path_pos=None, path_neg=None,
                 winSize=None, blockSize=None, blockStride=None, cellSize=None, nbins=None,
                 winStride=None, padding=None, scale=None):

        self.winStride = winStride
        self.padding = padding
        self.scale = scale
        self.cls = Classifier(path_pos=path_pos, path_neg=path_neg, winSize=winSize, blockSize=blockSize,
                              blockStride=blockStride, cellSize=cellSize, nbins=nbins)
        self.hogDescriptor = self.cls.hog
        self.cap = cv2.VideoCapture(0)
        self.add_model()

    def add_model(self):
        tree = ET.parse('./svm.xml')
        root = tree.getroot()
        # now this is really dirty, but after ~3h of fighting OpenCV its what happens :-)
        SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
        rho = float(root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text)
        svmvec = [float(x) for x in re.sub('\s+', ' ', SVs.text).strip().split(' ')]
        svmvec.append(-rho)
        pickle.dump(svmvec, open("./svm.pickle", 'wb'))
        svm = pickle.load(open("./svm.pickle", 'rb'))
        print(svm)
        self.hogDescriptor.setSVMDetector(np.array(svm))

    def run(self):
        print("RUNNING")
        while True:
            _, frame = self.cap.read()
            rects, weights = self.hogDescriptor.detectMultiScale(frame, winStride=self.winStride, padding=self.padding,
                                                                 scale=self.scale)
            print(rects)
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Detections", frame)
            cv2.waitKey(0)

DM = DetectMultiScale(path_pos="./train_pos_set", path_neg="./train_neg_set", winSize=(64,64), blockSize=(16,16),
                      blockStride=(8,8), cellSize=(8,8), nbins=9,
                      winStride=(8,8), padding=(8,8), scale=1.4)

DM.run()
