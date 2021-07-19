import numpy as np
import itertools

from numpy.core.fromnumeric import ndim, shape
from numpy.lib import RankWarning, pad
from numpy.lib.index_tricks import AxisConcatenator
from numpy.lib.shape_base import expand_dims

from RandomizedSelection import randomizedPartition

vec_add = lambda a,b : tuple([x+y for x,y in zip(a,b)])

def filter(img:np.ndarray, kernel:np.ndarray, step=1, pad_size=0):
    if 2 == kernel.ndim:
        return filter_2d(img,kernel, step, pad_size)
    img_pad = np.pad(img, [(pad_size,)]*kernel.ndim + [(0,)]*(img.ndim-kernel.ndim))
    o_idx = iterIdx(tuple([img_pad.shape[i]-kernel.shape[i]+1 for i in range(kernel.ndim)]), step)
    shape_out = tuple([int((2*pad_size+w-kw)/step + 1) for w, kw in zip(img.shape, kernel.shape)])
    shape_rem = tuple([img.shape[k] for k in range(kernel.ndim, img.ndim)])
    i_idx = iterIdx(shape_out)
    shape_out += shape_rem
    img_out = np.zeros(shape_out)
    ker = np.expand_dims(kernel, axis=tuple(range(kernel.ndim, img_pad.ndim))) if kernel.ndim < img_pad.ndim else kernel
    for i, o in zip(i_idx, o_idx):
        img_out[i] = np.sum((img_pad[tuple([range(x, x+w) for x, w in zip(o, kernel.shape)])]*ker).reshape((-1,) + shape_rem), axis=0)
    return img_out

def filter_2d(img:np.ndarray, kernel:np.ndarray, step=1, pad_size=0):
    img_pad = np.pad(img, [(pad_size,)]*kernel.ndim + [(0,)]*(img.ndim-kernel.ndim))
    shape_out = tuple([int((2*pad_size+w-kw)/step + 1) for w, kw in zip(img.shape, kernel.shape)])
    shape_rem = tuple([img.shape[k] for k in range(kernel.ndim, img.ndim)])
    shape_out += shape_rem
    img_out = np.zeros(shape_out)
    ker = np.expand_dims(kernel, axis=tuple(range(kernel.ndim, img_pad.ndim))) if kernel.ndim < img_pad.ndim else kernel
    for i, y0 in enumerate(range(0, img_pad.shape[0]-kernel.shape[0]+1, step)):
        for j, x0 in enumerate(range(0, img_pad.shape[1]-kernel.shape[1]+1, step)):
            img_out[i, j] = np.sum((img_pad[y0:y0+kernel.shape[0], x0:x0+kernel.shape[1]] * ker).reshape((-1,) + shape_rem), axis=0)
            # img_out[i, j] = np.sum((img_pad[tuple([range(x, x+w) for x, w in zip((y0, x0), kernel.shape)])] * ker).reshape((-1,) + shape_rem), axis=0)
    return img_out

# def padImg(img:np.ndarray, pad_size=None):
#     if pad_size is None:
#         return img
#     pad_width = [(w, w) for w in pad_size] + [(0, 0)]*(img.ndim-len(pad_size))
#     img_pad = np.pad(img, pad_width)
#     return img_pad

def iterIdx(shape, step=1):
    return itertools.product(*[range(0, i, step) for i in shape])

def generateGaussianKernel(ksize=3, sigma=0.8):
    kernel = np.zeros((ksize, ksize))
    origin = int(ksize / 2)
    for y in range(ksize):
        for x in range(ksize):
            kernel[y, x] = np.e**(-((x-origin)**2 + (y-origin)**2)/(2*sigma**2))
    kernel = np.floor(kernel / kernel[0, 0])
    kernel /= np.sum(kernel)
    return kernel

def generateLaplacianKernel():
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / 8
    return kernel

def gaussianFilter(img:np.ndarray, ksize=3, sigma=0.8):
    kernel = generateGaussianKernel(ksize, sigma)
    return filter(img, kernel, step=1, pad_size=1)

def generateGaussianVector(ksize=3, sigma=0.8):
    vec = np.zeros((ksize))
    origin = int(ksize / 2)
    for x in range(ksize):
        vec[x] = (np.e**(-(x-origin)**2))/(2*sigma**2)
    vec = np.floor(vec / vec[0])
    vec /= np.sum(vec)
    return vec

def fastGaussianFilter(img:np.ndarray, ksize=3, sigma=0.8):
    vec = generateGaussianVector(ksize, sigma)
    img_conv = np.pad(img, [(1,)]*2+[(0,)]*(img.ndim-2))
    img_conv = filter(img, vec.reshape(1, -1), step=1, pad_size=0)
    img_conv = filter(img, vec.reshape(-1, 1), step=1, pad_size=0)
    return img_conv

def laplacianFilter(img:np.ndarray):
    kernel = generateLaplacianKernel()
    return filter(img, kernel, step=1, pad_size=1)

def generateLaplacianOfGaussianKernel(ksize=3, sigma=0.8):
    kernel = np.zeros((ksize, ksize))
    origin = int(ksize / 2)
    for y in range(ksize):
        for x in range(ksize):
            kernel[y, x] = ((x-origin)**2 + (y-origin)**2 - 2*sigma**2)/(sigma**4) * np.e**(-((x-origin)**2 + (y-origin)**2)/(2*sigma**2))
    kernel = np.floor(kernel / kernel[0, 0])
    kernel /= np.sum(kernel)
    return kernel

def laplacianOfGaussianFilter(img:np.ndarray, ksize=3, sigma=0.8):
    kernel = generateLaplacianOfGaussianKernel(ksize, sigma)
    return filter(img, kernel, step=1, pad_size=1)

def fastLaplacianOfGaussianFilter(img:np.ndarray, ksize=3, sigma=0.8):
    kernel = generateLaplacianOfGaussianKernel(ksize, sigma)
    return filter(img, kernel, step=1, pad_size=1)

def separableFilter(img:np.ndarray, kernel:np.ndarray, step=1, pad_size=0):
    U, s, VT = np.linalg.svd(kernel)
    u, v = (s[0]**0.5)*U[:, [0]], (s[0]**0.5)*VT[[0], :]
    img_conv = np.pad(img, [(pad_size,)]*kernel.ndim+[(0,)]*(img.ndim-kernel.ndim))
    img_conv = filter(img_conv, v, step=step, pad_size=0)
    img_conv = filter(img_conv, u, step=step, pad_size=0)
    return img_conv

def summedAreaTable(img:np.ndarray):
    s = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            s[i, j] = (0 if i-1 < 0 else s[i-1, j]) + (0 if j-1 < 0 else s[i, j-1]) - (0 if i-1 < 0 or j-1 < 0 else (s[i-1, j-1])) + img[i, j]
    return s

areaComputation = lambda a, i0, j0, i1, j1 : a[i1, j1] - (0 if j0-1 < 0 else a[i1, j0-1]) - (0 if i0-1 < 0 else a[i0-1, j1]) + (0 if i0-1 < 0 or j0-1 < 0 else a[i0-1, j0-1])

# def areaComputation(a:np.ndarray, i0, j0, i1, j1):
#     s = a[i1, j1] - (0 if j0-1 < 0 else a[i1, j0-1]) - (0 if i0-1 < 0 else a[i0-1, j1]) + (0 if i0-1 < 0 or j0-1 < 0 else a[i0-1, j0-1])
#     return s

def meanFilter(img:np.ndarray, ksize):
    img_out = np.zeros_like(img)
    shape_rem = tuple([img.shape[k] for k in range(2, img.ndim)])
    r = int(ksize/2)
    # summed_area = summedAreaTable(img)
    # k = 1 / (2 * r + 1)**2
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # img_out[i, j] = areaComputation(summed_area, max(i-r, 0), max(j-r, 0), min(i+r, img.shape[0]-1), min(j+r, img.shape[1]-1))
            img_out[i, j] = np.mean((img[max(i-r, 0):min(i+r+1, img.shape[0]), max(j-r, 0):min(j+r+1, img.shape[1])]).reshape((-1,)+shape_rem), axis=0)
    return img_out


# a = np.arange(12).reshape(2, 2, 3)
# b = np.arange(48).reshape(2, 3, 4, 2)
# k_idx = iter_idx(a.shape)
# o_idx = iter_idx([b.shape[i] for i in range(a.ndim)], 1)
# for o in o_idx:
#     for i in k_idx:
#         print(b[vec_add(o, i)])
