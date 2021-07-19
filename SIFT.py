from typing import Tuple
import numpy as np

SIFT_MAX_INTERP_STEPS = 5
SIFT_IMG_BORDER = 1
FEATURE_MAX_D = 128
SIFT_ORI_HIST_BINS = 36
SIFT_ORI_PEAK_RATIO = 0.8
SIFT_DESCR_SCL_FCTR = 3.0
SIFT_DESCR_MAG_THR = 0.2

SIFT_INTVLS = 3
SIFT_SIGMA = 1.6
SIFT_CONTR_THR = 0.04
SIFT_CURV_THR = 10
SIFT_IMG_DBL = 1
SIFT_DESCR_WIDTH = 4
SIFT_DESCR_HIST_BINS = 8

class feature:
    def __init__(self, x=None, y=None, a=None, b=None, c=None, scl=None, ori=None, d=None, descr=None, type=None, category=None, img_pt=None, mdl_pt=None, ddata=None) -> None:
        self.x = x                           # /**< x coord */
        self.y = y                           # /**< y coord */
        self.a = a                           # /**< Oxford-type affine region parameter */
        self.b = b                           # /**< Oxford-type affine region parameter */
        self.c = c                           # /**< Oxford-type affine region parameter */
        self.scl = scl                         # /**< scale of a Lowe-style feature */
        self.ori = ori                         # /**< orientation of a Lowe-style feature */
        self.d = d                           # /**< descriptor length */
        self.descr = descr       # /**< descriptor */
        self.type = type                        # /**< feature type, OXFD or LOWE */
        self.category = category                    # /**< all-purpose feature category */
        self.img_pt = img_pt                      # /**< location in image */
        self.mdl_pt = mdl_pt                      # /**< location in model */
        self.ddata:detection_data = ddata if ddata else detection_data()               # /**< user-definable data */
    def copy(self):
        return feature(self.x, self.y,self.a, self.b, self.c, self.scl, self.ori, self.d, self.descr, self.type, self.category, self.img_pt, self.mdl_pt, self.ddata)

class detection_data:
    def __init__(self, r=None, c=None, octv=None, intvl=None, subintvl=None, scl_octv=None) -> None:
        self.r = r
        self.c = c
        self.octv = octv
        self.intvl = intvl
        self.subintvl = subintvl
        self.scl_octv = scl_octv


def sift_features(img:np.ndarray, intvls=SIFT_INTVLS, sigma=SIFT_SIGMA, contr_thr=SIFT_CONTR_THR, curv_thr=SIFT_CURV_THR, dbl=SIFT_IMG_DBL, descr_width=SIFT_DESCR_WIDTH, descr_hist_bins=SIFT_DESCR_HIST_BINS):
    # 1, generate gaussian pyramid
    octvs=int(np.log2(min(img.shape[0:2]))-3)
    gaussian_pyr = buildGaussianPyramid(img, octvs, intvls=intvls, sigma=sigma)
    # 2, generate difference of gaussian
    dog_pyr = buildDiffrenceOfGaussain(gaussian_pyr)
    # 3, detect extrema
    features = scaleSpaceExtrema(dog_pyr, octvs, intvls, contr_thr, curv_thr, sigma)
    # 4, calculate orientation
    feats = calculateFeatureOris(features, gaussian_pyr)
    # 5, calculate descriptors
    computeDescriptors(feats, gaussian_pyr, descr_width, descr_hist_bins)

    return feats


def buildGaussianPyramid(img:np.ndarray, octvs, intvls, sigma):
    gaussian_pyr = []
    k = pow(2, 1.0/intvls)
    sig = [0] * (intvls+3)
    sig[0] = sigma
    sig[1] = sigma * np.sqrt(k*k - 1)
    for i in range(2, len(sig)):
        sig[i] = sig[i-1] * k
    for o in range(octvs):
        octv_arr = []
        for i in range(intvls+3):
            if 0 == o and 0 == i:
                octv_arr.append(img)
            elif 0 == i:
                octv_arr.append(downsample(gaussian_pyr[-1][intvls]))
            else:
                octv_arr.append(gaussianBlur(octv_arr[-1], sig[i]))
        gaussian_pyr.append(octv_arr)

    return gaussian_pyr

def gaussianBlur(img:np.ndarray, sigma=0.8):
    w = upNearestOdd(6*sigma+1)
    kernel = generateGaussianKernel((w, w), sigma)
    return conv2d(img, kernel, step=1, pad_size=int((w-1)/2))

def generateGaussianKernel(ksize=(3, 3), sigma=0.8):
    kernel = np.zeros(ksize)
    h, w = ksize
    x0, y0 = int(w / 2), int(h / 2)
    for y in range(h):
        for x in range(w):
            kernel[y, x] = np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def conv2d(img:np.ndarray, kernel:np.ndarray, step=1, pad_size=0):
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

elewise_mul_as_int = lambda a,b : tuple([int(np.ceil(x*y)) for x,y in zip(a,b)])

def downsample(img:np.ndarray, rt=(0.5, 0.5)):
    shape_pre = elewise_mul_as_int(img.shape[0:len(rt)], rt)
    shape_rem = tuple(img.shape[len(rt):img.ndim])
    img_out = np.zeros((shape_pre+shape_rem))
    kh, kw = img.shape[0]/img_out.shape[0], img.shape[1]/img_out.shape[1]
    for i in range(img_out.shape[0]):
        for j in range(img_out.shape[1]):
            img_out[i, j] = np.average(img[int(np.floor(i*kh)):int(np.ceil((i+1)*kh)), int(np.floor(j*kw)):int(np.ceil((j+1)*kw))], axis=(0, 1))
    return img_out

def buildDiffrenceOfGaussain(gaussian_pyr):
    dog_pyr = [[octv[i+1]-octv[i] for i in range(len(octv)-1)] for octv in gaussian_pyr]
    return dog_pyr

def scaleSpaceExtrema(dog_pyr, octvs, intvls, contr_thr, curv_thr, sigma):
    features = []
    prelim_contr_thr = 0.5 * contr_thr / intvls
    for o in range(octvs):
        h, w = dog_pyr[o][0].shape[0:2]
        for i in range(1, intvls+1):
            for r in range(SIFT_IMG_BORDER, h-SIFT_IMG_BORDER):
                for c in range(SIFT_IMG_BORDER, w-SIFT_IMG_BORDER):
                    if np.abs(dog_pyr[o][i][r, c]) > prelim_contr_thr:
                        if isExtremum(dog_pyr, o, i, r, c):
                            feat = interpExtremum(dog_pyr, o, i, r, c, intvls, contr_thr, sigma)
                            if feat:
                                if not is_too_edge_like(dog_pyr, feat.ddata.octv, feat.ddata.intvl, feat.ddata.r, feat.ddata.c, curv_thr):
                                    features.append(feat)
    return features

def isExtremum(dog_pyr, o, i, r, c):
    return ((4 == np.argmin(dog_pyr[o][i][r-1:r+2, c-1:c+2])
            and dog_pyr[o][i][r,c] <= np.min(dog_pyr[o][i-1][r-1:r+2, c-1:c+2])
            and dog_pyr[o][i][r,c] <= np.min(dog_pyr[o][i+1][r-1:r+2, c-1:c+2]))
            or (4 == np.argmax(dog_pyr[o][i][r-1:r+2, c-1:c+2])
            and dog_pyr[o][i][r,c] >= np.max(dog_pyr[o][i-1][r-1:r+2, c-1:c+2])
            and dog_pyr[o][i][r,c] >= np.max(dog_pyr[o][i+1][r-1:r+2, c-1:c+2])))

def derive3d(dog_pyr, o, i, r, c):
    dx = (dog_pyr[o][i][r, c+1] - dog_pyr[o][i][r, c-1]) / 2
    dy = (dog_pyr[o][i][r+1, c] - dog_pyr[o][i][r-1, c]) / 2
    di = (dog_pyr[o][i+1][r, c] - dog_pyr[o][i-1][r, c]) / 2
    return np.array([[dx], [dy], [di]])

def hessian3d(dog_pyr, o, i, r, c):
    dxx = dog_pyr[o][i][r, c+1] + dog_pyr[o][i][r, c-1] - 2*dog_pyr[o][i][r, c]
    dyy = dog_pyr[o][i][r+1, c] + dog_pyr[o][i][r-1, c] - 2*dog_pyr[o][i][r, c]
    daa = dog_pyr[o][i+1][r, c] + dog_pyr[o][i-1][r, c] - 2*dog_pyr[o][i][r, c]
    dxy = (dog_pyr[o][i][r+1, c+1] + dog_pyr[o][i][r-1, c-1] - dog_pyr[o][i][r-1, c+1] - dog_pyr[o][i][r+1, c-1])/4.0
    dxa = (dog_pyr[o][i+1][r, c+1] + dog_pyr[o][i-1][r, c-1] - dog_pyr[o][i-1][r, c+1] - dog_pyr[o][i+1][r, c-1])/4.0
    dya = (dog_pyr[o][i+1][r+1, c] + dog_pyr[o][i-1][r-1, c] - dog_pyr[o][i-1][r+1, c] - dog_pyr[o][i+1][r-1, c])/4.0
    return np.array([[dxx, dxy, dxa], [dxy, dyy, dya], [dxa, dya, daa]])

def interpStep(dog_pyr, o, i, r, c):
    dD = derive3d(dog_pyr, o, i, r, c)
    H_inv = np.linalg.inv(hessian3d(dog_pyr, o, i, r, c))
    offset = -np.dot(H_inv, dD)
    return offset

def interpValue(dog_pyr, o, i, r, c, offset):
    dD = derive3d(dog_pyr, o, i, r, c)
    return dog_pyr[o][i][r, c] + 0.5*np.dot(dD.T, offset)[0, 0]

def interpExtremum(dog_pyr, o, i, r, c, intvls, contr_thr, sigma):
    for k in range(SIFT_MAX_INTERP_STEPS):
        offset = interpStep(dog_pyr, o, i, r, c)
        if (np.abs(offset) < 0.5).all():
            contr = interpValue(dog_pyr, o, i, r, c, offset)
            if np.abs(contr) < contr_thr/intvls:
                return None
            feat = feature()
            feat.ddata.r = r
            feat.ddata.c = c
            feat.ddata.octv = o
            feat.ddata.intvl = i
            feat.ddata.subintvl = offset[2, 0]
            feat.ddata.scl_octv = sigma * pow(2, (i+offset[2, 0])/intvls)
            feat.img_pt = ((c+offset[0, 0])*pow(2, o), (r+offset[1, 0])*pow(2, o))
            feat.scl = sigma * pow(2.0, o + (i+offset[2, 0])/intvls)
            return feat
        else:
            i += int(np.round(offset[2, 0]))
            r += int(np.round(offset[1, 0]))
            c += int(np.round(offset[0, 0]))
            h, w = dog_pyr[o][0].shape[0:2]
            if (i < 1) or (i > intvls) or (c < SIFT_IMG_BORDER) or (r < SIFT_IMG_BORDER) or (c >= (w - SIFT_IMG_BORDER)) or (r >= (h - SIFT_IMG_BORDER)):
                return None
    return None

def is_too_edge_like(dog_pyr, o, i, r, c, curv_thr):
    dxx = dog_pyr[o][i][r, c+1] + dog_pyr[o][i][r, c-1] - 2*dog_pyr[o][i][r, c]
    dyy = dog_pyr[o][i][r+1, c] + dog_pyr[o][i][r-1, c] - 2*dog_pyr[o][i][r, c]
    dxy = (dog_pyr[o][i][r+1, c+1] + dog_pyr[o][i][r-1, c-1] - dog_pyr[o][i][r-1, c+1] - dog_pyr[o][i][r+1, c-1])/4.0
    trH = dxx + dyy
    detH = dxx*dyy - dxy*dxy
    if detH <= 0:
        return True
    return (trH**2)/detH >= ((curv_thr+1)**2)/curv_thr

def upNearestOdd(a):
    b = round(a)
    if 0 == b%2:
        b += 1
    return b

def gradientMatrix(img:np.ndarray, r, c, rad):
    h, w = img.shape[0:2]
    r0 = 0 if r-rad < 0 else r-rad
    r1 = h if r+rad+1 > h else r+rad+1
    c0 = 0 if c-rad < 0 else c-rad
    c1 = w if c+rad+1 > w else c+rad+1
    win = img[r0:r1, c0:c1]
    dy = np.gradient(win, axis=0)
    dx = np.gradient(win, axis=1)
    grad_mat = np.zeros((2*rad+1, 2*rad+1, 2))
    grad_mat[r0+rad-r:r1+rad-r, c0+rad-c:c1+rad-c, 0] = (dy**2 + dx**2)**0.5
    grad_mat[r0+rad-r:r1+rad-r, c0+rad-c:c1+rad-c, 1] = np.arctan(dy/dx)%(2*np.pi)
    return grad_mat

def gradMat2Hist(grad_mat, mag_wgt, n):
    hist = [0] * n
    mag_mat = grad_mat[:, :, 0] * mag_wgt
    bin_idx = (grad_mat[:, :, 1]/(2*np.pi)*n).astype(np.uint)
    h, w = bin_idx.shape[0:2]
    for i in range(h):
        for j in range(w):
            hist[bin_idx[i, j]] += mag_mat[i, j]
    return hist

def smoothOriHist(grad_hist):
    n = len(grad_hist)
    g_hist = [0]*n
    for i in range(n):
        # g_hist[i] = (grad_hist[(i-2)%n]+grad_hist[(i+2)%n])/16 + (4*(grad_hist[(i-1)%n]+grad_hist[(i+1)%n]))/16 + (6*grad_hist[i])/16
        g_hist[i] = 0.25*(grad_hist[(i-1)%n]+grad_hist[(i+1)%n]) + 0.5*grad_hist[i]
    return g_hist

def hist2grad(hist, mag_thr):
    n = len(hist)
    grad = []
    for i, mag in enumerate(hist):
        if mag > mag_thr:
            l, r = (i-1)%n, (i+1)%n
            if hist[l] <= hist[i] <= hist[r]:
                dbin, peak = interpHistPeak(hist[l], hist[i], hist[r])
                bin = i + dbin
                bin = (bin+n) if bin < 0 else ((bin-n) if bin >= n else bin)
                grad.append((peak, bin*2*np.pi/n))
    return grad

def interpHistPeak(vl, vi, vr):
    return 0.5*(vl - vr) / (vl + vr - 2*vi), 0.125*(vr-vl)**2/(2*vi-vl-vr) + vi

def calculateFeatureOris(features, gaussian_pyr):
    feats = []
    feat:feature
    for feat in features:
        hist = oriHist(gaussian_pyr[feat.ddata.octv][feat.ddata.intvl],
                               feat.ddata.r, feat.ddata.c, SIFT_ORI_HIST_BINS,
                               int(np.round(3*1.5*feat.ddata.scl_octv)),
                               1.5*feat.ddata.scl_octv)
        hist = smoothOriHist(hist)
        grad = hist2grad(hist, np.max(hist)*SIFT_ORI_PEAK_RATIO)
        for _, ori in grad:
            new_feat = feat.copy()
            new_feat.ori = ori
            feats.append(new_feat)
    return feats


def oriHist(img:np.ndarray, r, c, n, rad, sigma):
    grad_mat = gradientMatrix(img, r, c, rad)
    mag_wgt = generateGaussianKernel(ksize=grad_mat.shape[0:2], sigma=sigma)
    hist = gradMat2Hist(grad_mat, mag_wgt, n)
    return hist

def computeDescriptors(feats, gaussian_pyr, d, n):
    feat:feature
    for feat in feats:
        hist = descrHist(gaussian_pyr[feat.ddata.octv][feat.ddata.intvl], feat.ddata.r, feat.ddata.c, feat.ori, feat.ddata.scl_octv, d, n)
        feat.descr = hist2descr(hist)

def descrHist(img:np.ndarray, r, c, ori, scl, d, n):
    hist = np.zeros((d, d, n))
    cos_t, sin_t = np.cos(ori), np.sin(ori)
    bins_per_rad = n / (2*np.pi)
    exp_denom = 2 * (0.5*d)**2
    hist_width = SIFT_DESCR_SCL_FCTR * scl
    rad = hist_width * 2**0.5 * (d+1) * 0.5 + 0.5
    h, w = 2*int(rad)+1, 2*int(rad)+1
    grad_mat = gradientMatrix(img, r, c, int(rad))
    coor_ori = np.array([[i, j] for i in range(-int(rad), int(rad)+1) for j in range(-int(rad), int(rad)+1)])
    coor_rot = (np.dot(coor_ori, np.array([[cos_t, sin_t], [-sin_t, cos_t]]))/hist_width).reshape((h, w, 2))
    coor_ori = coor_ori.reshape((h, w, 2))
    bin = coor_rot + d/2.0 - 0.5
    for i in range(h):
        for j in range(w):
            if -1.0 < bin[i, j, 0] < d and -1.0 < bin[i, j, 1] < d and grad_mat[i, j, 0] > 0:
                grad_ori = (grad_mat[i, j, 1] - ori) % (2*np.pi)
                grad_mag = grad_mat[i, j, 0] * np.exp(-(coor_rot[i, j, 0]**2 + coor_rot[i, j, 1]**2)/exp_denom)
                rbin, cbin, obin = bin[i, j, 0], bin[i, j, 1], grad_ori * bins_per_rad
                interpHistEntry(hist, rbin, cbin, obin, grad_mag)
    return hist


def interpHistEntry(hist, rbin, cbin, obin, mag):
    h, w, n = hist.shape
    r0, c0, o0 = int(np.floor(rbin)), int(np.floor(cbin)), int(np.floor(obin))
    dr, dc, do = rbin - r0, cbin - c0, obin - o0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if 0 <= r0+i < h and 0 <= c0+j < w:
                    hist[r0+i, c0+j, (o0+k)%n] = mag * dr**i * (1-dr)**(1-i) * dc**j * (1-dc)**(1-j) * do**k * (1-do)**(1-k)

def hist2descr(hist:np.ndarray):
    descr = hist.flatten()
    descr = descr / ((np.sum(descr**2))**0.5)
    descr[descr>SIFT_DESCR_MAG_THR] = SIFT_DESCR_MAG_THR
    descr = descr / ((np.sum(descr**2))**0.5)
    return descr


