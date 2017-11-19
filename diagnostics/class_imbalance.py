# Run this with: python class_imbalance.py

from __future__ import unicode_literals, print_function
from network.get_class_weights import approximate_class_frequencies


def main():

    n_samples = 100
    f = approximate_class_frequencies(sample_ts=n_samples, hemisphere="rh")

    print("Class frequencies")
    print(f)

    print("Class frequencies in list (that you can copy paste directly into config.py)")
    print(f.values())


if __name__ == '__main__':
    main()
