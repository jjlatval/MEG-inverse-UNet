# Run this with: python class_imbalance.py

from __future__ import unicode_literals, print_function, division

import six

from config import HEMISPHERE
from data_provider.data_provider import GeneratorDataProvider


def main():

    g = GeneratorDataProvider("train", HEMISPHERE)
    t = g.t_steps
    n = g.n_vertices
    a = [0] * n

    for i in range(0, t):
        x, y = six.next(g)
        a[y] += 1

    f = [x / t for x in a]
    print("Class occurences")
    print(a)
    print("Class frequencies")
    print(f)


if __name__ == '__main__':
    main()
