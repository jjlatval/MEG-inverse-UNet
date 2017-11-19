# General util functions

from __future__ import division, unicode_literals
import math
import numpy as np


def factors(n):
    return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def most_square_shape(n):
    f = np.array(list(factors(n)))
    x = find_nearest(f, math.sqrt(n))
    y = n / x
    return int(x), int(y)


def get_nearest_root(length):
    root = int(math.floor(math.sqrt(length)))
    return root


def get_nearest_even_root(length):
    root = int(math.floor(math.sqrt(length)))
    if root < 10:
        raise ValueError("The size of the target array is too small.")
    if not root % 2 == 0:
        root -= 1
    return root
