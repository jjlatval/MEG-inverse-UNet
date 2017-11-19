# Resolution metrics are used for comparing the network prediction to inverse solutions.

from __future__ import division, unicode_literals
import math
import numpy as np


def dipole_localization_error(array1, ground_truth, array2=None, border_pixels=None, zero_class_one_hot_encoded=[1., 0.]):
    """
    Returns dipole localization error given
    :param array1: numpy array of shape [n, nx, ny, channels]
    :param ground_truth: numpy array of shape [n, nx, ny, classes]
    :param array2: numpy array of shape [n, nx, ny, classes]
    :param border_pixels: int
        Amount of empty pixels surrounding array2 in order to make its nx, ny equal to array1
    :return dle_array1, dle_array2
    """


    channels = array1.shape[-1]
    dle1 = np.zeros(ground_truth.shape[2], channels)

    for y in range(0, ground_truth.shape[2]):
        activation_locations = np.where(array1[0, :, y, :] != zero_class_one_hot_encoded)  # it is a tuple with a value of e.g. (array([407, 407]), array([0, 1]))
        for c in range(0, channels):
        dle1[y, c] = math.sqrt((np.argmax(array1[0, :, y, c]) - activation_locations) ** 2)

    if array2:
        if not array2.shape[1] + border_pixels * 2 == array1.shape[1]:
            raise ValueError("Arrays 1 and 2 do not match in x axis with border pixels: %s" % border_pixels)
        if not array2.shape[2] + border_pixels * 2 == array1.shape[2]:
            raise ValueError("Arrays 1 and 2 do not match in y axis with border pixels: %s" % border_pixels)
