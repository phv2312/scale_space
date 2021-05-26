import cv2
import numpy as np
import matplotlib.pyplot as plt
import im_utils
import contour_utils
import scale_space_utils

def sampling(im_path, max_scale=32):
    np_image = im_utils.imread(im_path, mode='L')
    skeleton = im_utils.thin(np_image)
    h,w = np_image.shape[:2]

    # extract contour
    contour = cv2.findContours(skeleton, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[1][0]
    contour = contour_utils.normalize_contour(contour)

    # smooth
    n_point = len(contour)
    scales  = range(1, max_scale)
    n_scale = len(scales)  # need +1
    curvature_matrix = np.zeros(shape=(n_point, n_scale + 1), dtype=np.float)
    scale_space_repr = []

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    ax1 = fig.axes[0]
    for scale_id, scale in enumerate(scales):
        contour_smooth = scale_space_utils.smooth(contour, scale)
        contour_smooth = contour_utils.normalize_contour(contour_smooth)

        curvature = contour_utils.calc_curvature(contour_smooth)
        extreme_ids = contour_utils.find_local_extreme(curvature)
        extreme_ids_2 = np.where(np.abs(curvature) > 0.2)[0]
        extreme_ids = np.intersect1d(extreme_ids_2, extreme_ids, assume_unique=True)

        # save
        curvature_matrix[np.arange(n_point), scale] = curvature
        for extreme_id in extreme_ids:
            scale_space_repr += [(extreme_id, scale)]

        plt.cla()
        ax1.scatter(contour_smooth[:, 0, 0], h - contour_smooth[:, 0, 1])
        ax1.scatter(contour_smooth[extreme_ids, 0, 0], h - contour_smooth[extreme_ids, 0, 1],
                    s=60, c='red', marker='x')
        plt.text(0.87, 0.92, 'scale: %d' % scale, horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, fontsize='x-large')
        plt.draw()
        plt.pause(0.1)

    scale_space_repr = np.array(scale_space_repr)
    extreme_points = scale_space_utils.group(scale_space_repr, curvature_matrix)

    color_list = ['red', 'green', 'blue', 'yellow']
    plt.figure('localizing point')
    plt.scatter(contour[:, 0, 0], h - contour[:, 0, 1])
    for count, extreme_point in enumerate(extreme_points):
        arclen, scale = extreme_point[0]
        plt.scatter(contour[arclen, 0, 0], h - contour[arclen, 0, 1], s=150,
                    c=color_list[count % len(color_list)], marker='x')
    plt.show()


if __name__ == '__main__':
    im_path = "./data/contour1.png"
    sampling(im_path, max_scale=32)