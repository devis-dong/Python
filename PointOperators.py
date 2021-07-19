import numpy as np


def gainAndBias(img, a, b):
    return a * img + b

def dyadic(img0, img1, alpha):
    return (1 - alpha) * img0 + alpha * img1

def gammaCorrect(img, gamma):
    return img ** (1.0/gamma)

def histogramEqualization(img_gray:np.ndarray):
    img_equalized = np.zeros_like(img_gray)
    f = histgramEqualizationFunction(img_gray)
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            img_equalized[i, j] = f[img_gray[i, j]]
    return img_equalized

def histogram(img_gray:np.ndarray):
    num_pixel = np.zeros(256)
    h, w = img_gray.shape
    for i in range(h):
        for j in range(w):
            num_pixel[img_gray[i, j]] += 1
    return num_pixel

def cumHistogram(img_gray:np.ndarray):
    hist = histogram(img_gray)
    cum_hist = np.zeros(len(hist))
    cum_hist[0] = hist[0]
    for i in range(1, len(hist)):
        cum_hist[i] = cum_hist[i-1] + hist[i]
    return cum_hist

def histgramEqualizationFunction(img_gray:np.ndarray):
    cum_hist = cumHistogram(img_gray)
    return cum_hist / np.max(cum_hist) * 255

def blockHistgramEqualization(img_gray:np.ndarray, block_size:tuple):
    img_equalized = np.zeros_like(img_gray)
    h, w = img_gray.shape
    kh, kw = block_size
    for t in range(0, h, kh):
        for l in range(0, w, kw):
            img_equalized[t:min(t+kh, h), l:min(l+kw, w)] = histogramEqualization(img_gray[t:min(t+kh, h), l:min(l+kw, w)])
    return img_equalized

def blockHistogramEqualizationFunction(img_gray:np.ndarray, block_size:tuple):
    h, w = img_gray.shape
    kh, kw = block_size
    tiles_h, tiles_w = int(h / kh), int(w / kw)
    f = np.zeros((tiles_h, tiles_w, 256))
    for i, t in enumerate(range(0, h, kh)):
        for j, l in enumerate(range(0, w, kw)):
            f[i, j] = histgramEqualizationFunction(img_gray[t:min(t+kh, h), l:min(l+kw, w)])
    return f

def locallyAdaptiveHistogramEqualization(img_gray:np.ndarray, block_size:tuple):
    f = blockHistogramEqualizationFunction(img_gray, block_size)
    img_rt = np.zeros_like(img_gray)
    h, w = img_gray.shape
    kh, kw = block_size
    half_kh, half_kw = int(kh/2), int(kw/2)
    th, tw, _ = f.shape
    for i, y0 in enumerate(range(0, h, kh)):
        for j, x0 in enumerate(range(0, w, kw)):
            for y in range(y0, min(y0+kh, h)):
                for x in range(x0, min(x0+kw, w)):
                    I = img_gray[y, x]
                    i0, i1, t = (max(i-1, 0), i, (y-y0)/kh+0.5) if y < y0+half_kh else (i, min(i+1, th-1), (y-y0)/kh-0.5)
                    j0, j1, s = (max(j-1, 0), j, (x-x0)/kw+0.5) if x < x0+half_kw else (j, min(j+1, tw-1), (x-x0)/kw-0.5)
                    img_rt[y, x] = int((1-s)*(1-t)*f[i0, j0, I] + s*(1-t)*f[i0, j1, I] + (1-s)*t*f[i1, j0, I] + s*t*f[i1, j1, I])
    # for y in range(h):
    #     for x in range(w):
    #         I = img_gray[y, x]
    #         s, t = x/kw - 0.5, y/kh - 0.5
    #         i0, j0 = max(int(t), 0), max(int(s), 0)
    #         i1, j1 = min(i0+1, th-1), min(j0+1, tw-1)
    #         img_rt[y, x] = int((1-s)*(1-t)*f[i0, j0, I] + s*(1-t)*f[i0, j1, I] + (1-s)*t*f[i1, j0, I] + s*t*f[i1, j1, I])
    return img_rt


