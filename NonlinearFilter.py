import numpy as np
import RandomizedSelection as rs

def medianFilter(img:np.ndarray, ksize=3):
    img_out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            win = img[max(i-int(ksize/2), 0):min(i+int(ksize/2), img.shape[0]-1)+1, max(j-int(ksize/2), 0):min(j+int(ksize/2), img.shape[1]-1)+1]
            if img.ndim > 2:
                for c in range(img.shape[2]):
                    img_out[i, j, c] = rs.median(win[:, :, c].flatten())
            else:
                img_out[i, j] = rs.median(win.flatten())
            
    return img_out

def weightedMedianFilter(img:np.ndarray, weights:np.ndarray, ksize=3):
    img_out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            win = img[max(i-int(ksize/2), 0):min(i+int(ksize/2), img.shape[0]-1)+1, max(j-int(ksize/2), 0):min(j+int(ksize/2), img.shape[1]-1)+1]
            wts = weights[max(i-int(ksize/2), 0):min(i+int(ksize/2), weights.shape[0]-1)+1, max(j-int(ksize/2), 0):min(j+int(ksize/2), weights.shape[1]-1)+1]
            if img.ndim > 2:
                for c in range(img.shape[2]):
                    img_out[i, j, c] = rs.weightedMedian(win[:, :, c].flatten(), wts.flatten())
            else:
                img_out[i, j] = rs.weightedMedian(win.flatten(), wts.flatten())
            
    return img_out

