# Run this file to train the network: python train_network.py

from __future__ import unicode_literals

from config import HEMISPHERE, NETWORK_TYPE
from network.network_caller import NetworkCaller


def main():

    print("Resuming neural network training for hemisphere: %s" % HEMISPHERE)

    if NETWORK_TYPE == 'unet':
        net = NetworkCaller(HEMISPHERE)
        net.resume_training()

    else:
        raise ValueError("Network type %s not understood" % NETWORK_TYPE)


if __name__ == "__main__":
    main()
