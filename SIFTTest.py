import os
import numpy as np
import cv2 as cv
import time

import SIFT

print("current working space:", os.getcwd())
print('running ......')

# img = cv.cvtColor(cv.resize(cv.imread('../Data/imgs/apple1.jpg'), (256, 256)), cv.COLOR_BGR2GRAY)
# time0 = time.time()
# feats = SIFT.sift_features(img)
# print("sift_features:", time.time()-time0)
# kpts = [cv.KeyPoint(feat.img_pt[0], feat.img_pt[1], 1) for feat in feats]
# cv.imshow('sift features 0', cv.drawKeypoints(img, kpts, img, color=(0,0,255)))

img = cv.cvtColor(cv.resize(cv.imread('../Data/imgs/apple1.jpg'), (512, 512)), cv.COLOR_BGR2GRAY)
time0 = time.time()
feats = SIFT.sift_features(img)
print("sift_features:", time.time()-time0)
kpts = [cv.KeyPoint(feat.img_pt[0], feat.img_pt[1], 1) for feat in feats]
cv.imshow('sift features 1', cv.drawKeypoints(img, kpts, img, color=(0,0,255)))

img = cv.cvtColor(cv.resize(cv.imread('../Data/imgs/apple1.jpg'), (512, 512)), cv.COLOR_BGR2GRAY)
cvsift = cv.SIFT_create()
time0 = time.time()
kp, des = cvsift.detectAndCompute(img, None)
print("cv sift_features:", time.time()-time0)
cv.imshow('cv sift features 0', cv.drawKeypoints(img, kp, img, color=(0,0,255)))

cv.waitKey(0)

print('... done!!!')