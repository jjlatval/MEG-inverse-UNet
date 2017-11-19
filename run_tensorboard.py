# -*- coding: utf-8 -*-

# Use Tensorboard to go through the metrics of the neural network training
# Usage: python run_tensorboard.py

from __future__ import unicode_literals
import os
from config import WEIGHT_PATH


def main():

    os.system('tensorboard --logdir=' + WEIGHT_PATH)


if __name__ == "__main__":
    main()
