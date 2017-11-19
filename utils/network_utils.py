from __future__ import division
import numpy as np
import math
from config import N_CLASSES


def find_patch_size(length):
    """
    Finds two closest integers that when multiplied produce the input. In case of a prime number
    the function will return (1, prime)
    :param length: integer
    :return: tuple(integer, integer)
    """
    root = math.sqrt(length)
    if root.is_integer():
        return int(root), int(root)
    ceil = int(round(root))
    for i in range(ceil)[::-1]:
        if length % i == 0:
            return int(i), int(length / i)


def get_class_frequencies(generator, n_classes=N_CLASSES):
    """
    Gets class frequencies from a generator
    :param generator: generator object
        Generator in question that consists of n_classes distinct integers
    :param n_classes: integer
        Number of classes
    :return: numpy array
        Array of frequencies per class
    """
    freqs = np.zeros(N_CLASSES)
    for x in generator:
        freqs += np.bincount(x.astype(int), minlength=n_classes).astype(np.float64)
    return freqs


def calculate_confusion_matrices(conf, zero_index):
    """
    Calculates confusion matrix accuracies for both including and excluding zero value class
    :param conf: numpy array
        A regular confusion matrix of N_CLASSES x N_CLASSES
    :param zero_index: integer
        Index of zero class values
    :return: accuracy1, accuracy2
        Accuracy1 = Normal confusion matrix accuracy score including zero classes
        Accuracy2 = Confusion matrix accuracy score without zero classes
    """
    conf_sum = conf.sum()
    if conf_sum == 0:
        return np.nan, np.nan
    else:
        acc = conf.diagonal().sum() / conf_sum
    correct_zeros = conf.diagonal()[zero_index]
    all_zeros = conf[zero_index:zero_index + 1, zero_index].sum() + \
                conf[zero_index:, zero_index:zero_index + 1].sum()
    if all_zeros == conf_sum:
        acc2 = np.nan
    else:
        acc2 = (conf.diagonal().sum() - correct_zeros) / (conf_sum - all_zeros)
    return acc, acc2
