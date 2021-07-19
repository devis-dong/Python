import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import zeros_like
from numpy.lib.index_tricks import nd_grid


def binarize(img:np.ndarray, threshold, up_val):
    img_out = np.zeros_like(img)
    img_out[img>=threshold] = up_val
    return img_out

def dilate(img:np.ndarray, ksize):
    return threshMorphing(img, ksize, 1)

def erode(img:np.ndarray, ksize):
    return threshMorphing(img, ksize, ksize**2)

def majority(img:np.ndarray, ksize):
    return threshMorphing(img, ksize, 0.5 * ksize**2)

def threshMorphing(img:np.ndarray, ksize, threshold):
    img_out = np.zeros_like(img)
    r = int(ksize/2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_out[i, j] = 0 if (np.sum((img[max(0, i-r):min(i+r+1, img.shape[0]), max(0, j-r):min(j+r+1, img.shape[1])]) > 0) < threshold) else 255
    return img_out

def opening(img:np.ndarray, ksize):
    return dilate(erode(img, ksize), ksize)

def closing(img:np.ndarray, ksize):
    return erode(dilate(img, ksize), ksize)

def cityBlockDistance(img:np.ndarray):
    dist = np.zeros_like(img).astype(float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 0 != img[i, j]:
                dist[i, j] = 1 + min(np.Infinity if i-1 < 0 else dist[i-1, j], np.Infinity if j-1 < 0 else dist[i, j-1])
    for i in range(img.shape[0]-1, -1, -1):
        for j in range(img.shape[1]-1, -1, -1):
            if 0 != img[i, j]:
                dist[i, j] = min(dist[i, j], 1 + min(np.Infinity if i+1 >= img.shape[0] else dist[i+1, j], np.Infinity if j+1 >= img.shape[1] else dist[i, j+1]))
    return dist

# def connectedComponents(img:np.ndarray, gap=32):
#     comp_idx = zeros_like(img)
#     comp_cnt = 0 
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             dist = [np.Infinity if i-1 < 0 else np.abs(img[i,j]-img[i-1,j]), np.Infinity if j-1 < 0 else np.abs(img[i,j]-img[i, j-1]), np.Infinity if (i-1 < 0 or j-1 < 0) else np.abs(img[i, j]-img[i-1, j-1])]
#             idx = np.argmin(dist)
#             if dist[idx] >= gap:
#                 comp_idx[i, j] = comp_cnt
#                 comp_cnt += 1
#             else:
#                 comp_idx[i, j] = comp_idx[i-1, j] if 0 == idx else (comp_idx[i, j-1] if 1 == idx else comp_idx[i-1, j-1])
#     return comp_cnt, comp_idx


