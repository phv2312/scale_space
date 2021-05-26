import numpy as np
from scipy.ndimage import convolve1d
from scipy.spatial.distance import cdist

def gaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1).astype(np.float)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / (sigma * np.sqrt(2 * np.pi))
    phi_x = phi_x / phi_x.sum()

    return phi_x


def convolve(contour, weights, radius):
    """
    Convolve 1 contour with a kernel (weights), 1d convolution

    :param contour: (N,1,2)
    :param weights: (N,), gauss distribution
    :param radius:
    :return:
    """
    contour = contour[:,0,:]
    contour_insert_first = contour[-radius:]
    contour_insert_last  = contour[:(radius+1)]

    contour_new = np.concatenate([contour_insert_first, contour, contour_insert_last])
    contours_x = convolve1d(contour_new[:, 0], weights)[radius:-(radius+1)]
    contours_y = convolve1d(contour_new[:, 1], weights)[radius:-(radius+1)]

    contours_out = np.stack([contours_x, contours_y], axis=1)

    return contours_out[:, None, :]


def scale_contour(contours, scale_factor=4):
    """
    Scale, means smoothing the contour by convolve the original contour with a gaussian kernel

    :param contours:
    :param scale_factor:
    :return:
    """
    n_point = len(contours)
    radius = n_point // 2

    weights = gaussian_kernel1d(sigma=scale_factor, radius=radius)
    return convolve(contours, weights, radius=radius)


def normalize_contour(contour):
    """
    Normalize contour to the range(0,1)

    :param contour: keep the original shape
    :return:
    """

    contour_float = contour.astype(np.float)
    contour_x = contour_float[:,0,0]
    contour_x = (contour_x - contour_x.min()) / (contour_x.max() - contour_x.min())

    contour_y = contour_float[:,0,1]
    contour_y = (contour_y - contour_y.min()) / (contour_y.max() - contour_y.min())

    contour_new = np.stack([contour_x, contour_y], axis=1)[:,None,:]
    return contour_new


def calc_curvature(closed_contour, original_shape=(-1,-1)):
    """
    Calculate curvature of closed contour

    :param closed_contour:
    :param original_shape:
    :return:
    """

    assert len(closed_contour) > 2

    contour_float = closed_contour.astype(np.float)[:, 0, :]
    x = np.hstack([contour_float[-2:,0], contour_float[:,0], contour_float[:2,0]])
    y = np.hstack([contour_float[-2:,1], contour_float[:,1], contour_float[:2,1]])

    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    dx_dtt = np.gradient(dx_dt)
    dy_dtt = np.gradient(dy_dt)
    curvature = (dy_dtt * dx_dt - dx_dtt * dy_dt) / np.power(dx_dt * dx_dt + dy_dt * dy_dt, 1.5)

    return curvature[2:-2]


def find_local_extreme(curvatures):
    n_p = len(curvatures)
    ids = []

    for i in range(2, n_p - 1):
        if (abs(curvatures[i]) > abs(curvatures[i - 1]) and abs(curvatures[i]) > abs(curvatures[i + 1]) and
                abs(curvatures[i]) > abs(curvatures[i - 2]) and abs(curvatures[i]) > abs(curvatures[i + 2])):
            ids += [i]

    return ids