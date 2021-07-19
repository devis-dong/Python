from matplotlib.pyplot import triplot
import numpy as np
import math

#multipy elementwise
elewise_mul_as_int = lambda a,b : tuple([int(np.ceil(x*y)) for x,y in zip(a,b)])

def linear(f0, f1, r=0.5):
    return (1-r)*f0 + r*f1

def bilinear(Q11, Q12, Q21, Q22, ry=0.5, rx=0.5):
    return np.dot([[1-rx, rx]], [[Q11, Q12], [Q21, Q22]]).dot([[1-ry], [ry]])

def push(img:np.ndarray, kernel):
    return None

def pull(img:np.ndarray, kernel):
    return None

def sample(img:np.ndarray, rt=(2, 2)):
    shape_pre = elewise_mul_as_int(img.shape[0:len(rt)], rt)
    shape_rem = tuple(img.shape[len(rt):img.ndim])
    img_out = np.zeros((shape_pre+shape_rem))
    kh, kw = img.shape[0]/img_out.shape[0], img.shape[1]/img_out.shape[1]
    for i in range(img_out.shape[0]):
        for j in range(img_out.shape[1]):
            img_out[i, j] = np.average(img[int(np.floor(i*kh)):int(np.ceil((i+1)*kh)), int(np.floor(j*kw)):int(np.ceil((j+1)*kw))], axis=(0, 1))
    return img_out
