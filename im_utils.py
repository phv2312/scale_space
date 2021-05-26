import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def imread(im_path, mode='RGB'):
    """
    Read image numpy.ndarray
    :param im_path:
    :param mode:
    :return:
    """
    return np.array(Image.open(im_path).convert(mode))


def thin(np_image):
    _, threshold = cv2.threshold(np_image, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    dilated_image = cv2.dilate(threshold, kernel=np.ones(shape=(3, 3), dtype=np.int))
    dilated_image = (dilated_image / 255).astype(np.uint8)
    skeleton = skeletonize(dilated_image).astype(np.uint8)

    return skeleton


def imshow(im):
    plt.imshow(im)
    plt.show()