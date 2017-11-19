# This file visualizes simulated data
# Run this with: python visualize_simulated_data.py

from __future__ import print_function, unicode_literals

import matplotlib.pyplot as plt

from data_provider.data_provider import DataProvider


def main():

    data = DataProvider()

    # plt.plot(data.train_stc.times, data.train_stc.data[::100, :].T)
    plt.plot(data.stc.times, data.stc.data[:, :].T)
    plt.xlabel('time (ms)')
    plt.ylabel('Source amplitude')
    plt.title('train_stc dataset')
    plt.show()

if __name__ == '__main__':
    main()
