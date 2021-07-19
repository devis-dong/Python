import numpy as np


def translate(img:np.ndarray, tx, ty):
    h, w = img.shape[0:2]
    T_inv = np.array([[1, 0, -tx], [0, 1, -ty], [0, 0, 1]])
    img_out = np.zeros_like(img)
    for y_prime in range(h):
        for x_prime in range(w):
            x, y, _ = np.dot(T_inv, [[x_prime], [y_prime], [1]]).astype(np.int)[:, 0]
            img_out[y_prime, x_prime] = img[y, x] if (0 <= y < h and  0 <= x < w) else 0
    return img_out

def rotate(img:np.ndarray, degree=90, x0=None, y0=None):
    theta = np.pi * degree / 180
    h, w = img.shape[0:2]
    x0, y0 = w/2 if x0 is None else x0, h/2 if y0 is None else y0
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    R_inv = np.array([[cos_theta, sin_theta, x0*(1-cos_theta)-y0*sin_theta], [-sin_theta, cos_theta, y0*(1-cos_theta)+x0*sin_theta], [0, 0, 1]])
    img_out = np.zeros_like(img)
    for y_prime in range(h):
        for x_prime in range(w):
            x, y, _ = np.dot(R_inv, [[x_prime], [y_prime], [1]]).astype(np.int)[:, 0]
            img_out[y_prime, x_prime] = img[y, x] if (0 <= y < h and  0 <= x < w) else 0
    return img_out

def EuclideanTransform(img:np.ndarray, degree, tx=0, ty=0, x0=None, y0=None):
    theta = np.pi * degree / 180
    h, w = img.shape[0:2]
    x0, y0 = w/2 if x0 is None else x0, h/2 if y0 is None else y0
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    # T_inv = np.array([[1, 0, -tx], [0, 1, -ty], [0, 0, 1]])
    # R_inv = np.array([[cos_theta, sin_theta, x0*(1-cos_theta)-y0*sin_theta], [-sin_theta, cos_theta, y0*(1-cos_theta)+x0*sin_theta], [0, 0, 1]])
    R_inv_T_inv = np.array([[cos_theta, sin_theta, -tx*cos_theta - tx*sin_theta + x0*(1-cos_theta)-y0*sin_theta],
                            [-sin_theta, cos_theta, tx*sin_theta - ty*cos_theta + y0*(1-cos_theta)+x0*sin_theta],
                            [0, 0, 1]])
    img_out = np.zeros_like(img)
    for y_prime in range(h):
        for x_prime in range(w):
            # x, y, _ = np.dot(R_inv.dot(T_inv), [[x_prime], [y_prime], [1]]).astype(np.int)[:, 0]
            x, y, _ = np.dot(R_inv_T_inv, [[x_prime], [y_prime], [1]]).astype(np.int)[:, 0]
            img_out[y_prime, x_prime] = img[y, x] if (0 <= y < h and  0 <= x < w) else 0
    return img_out

def similarityTransform(img:np.ndarray, degree, x0=None, y0=None, tx=0, ty=0, s=1):
    theta = np.pi * degree / 180
    h, w = img.shape[0:2]
    x0, y0 = w/2 if x0 is None else x0, h/2 if y0 is None else y0
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    # T_inv = np.array([[1, 0, -tx], [0, 1, -ty], [0, 0, 1]])
    # R_inv = np.array([[cos_theta, sin_theta, x0*(1-cos_theta)-y0*sin_theta], [-sin_theta, cos_theta, y0*(1-cos_theta)+x0*sin_theta], [0, 0, 1]])
    S_R_inv_T_inv = np.array([[cos_theta/s, sin_theta/s, (-tx*cos_theta - tx*sin_theta + x0*(1-cos_theta)-y0*sin_theta - 1)/s],
                            [-sin_theta/s, cos_theta/s, (tx*sin_theta - ty*cos_theta + y0*(1-cos_theta)+x0*sin_theta - 1)/s],
                            [0, 0, 1]])
    img_out = np.zeros_like(img)
    for y_prime in range(h):
        for x_prime in range(w):
            # x, y, _ = np.dot(R_inv.dot(T_inv), [[x_prime], [y_prime], [1]]).astype(np.int)[:, 0]
            x, y, _ = np.dot(S_R_inv_T_inv, [[x_prime], [y_prime], [1]]).astype(np.int)[:, 0]
            img_out[y_prime, x_prime] = img[y, x] if (0 <= y < h and  0 <= x < w) else 0
    return img_out
