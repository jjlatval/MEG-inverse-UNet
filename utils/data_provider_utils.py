# data_provider related utils

from __future__ import division, unicode_literals
import numpy as np


def init_one_value_vector(length, value, data_type=np.float):
    """
    Initialize a zero vector with zero values
    :param length: integer
        Desired vector length
    :param value: integer | float
        Desired value to fill the vector with
    :param data_type: np.float | np.int
        Desired target type of data for the vector
    :return:
    """
    return np.full(length, value).astype(data_type)
