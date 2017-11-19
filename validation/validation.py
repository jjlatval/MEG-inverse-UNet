# This file contains validation functions for network calls.
from __future__ import unicode_literals, print_function
import platform
import multiprocessing
import re


def validate_cpu_threads(n_cpu):
    if n_cpu < 1:
        print("CPU_THREADS has to be at least 1. Setting CPU_THREADS to 1.")
        return 1
    if platform.system() == 'Darwin':
        print("Cannot set CPU_THREADS higher than 1 for MacOS. Setting CPU_THREADS to 1.")
        return 1  # NOTE: There may be some issues with multithreading when simulating data in MacOS
    return min(multiprocessing.cpu_count(), n_cpu)


def validate_spacing(spacing):
    if not spacing or re.compile('([i-o])\w{2}\d').match(spacing) is None:
        print("Incorrect SPACING %s. Setting SPACING to ico3.")
        return 'ico3'
    return spacing


def validate_channels_and_ground_truth(channels, ground_truth):
    if len(channels) < 1:
        print("There needs to be some input CHANNELS. Setting CHANNELS to ['mne', 'sloreta', 'dspm'].")
        channels = ['mne', 'sloreta', 'dspm']
    if len(ground_truth) != 1:
        print("There needs to be exactly one GROUND_TRUTH. Setting GROUND_TRUTH to ['stc'].")
        ground_truth = ['stc']
    return channels, ground_truth, len(channels)


def validate_n_classes(n_classes):
    if n_classes < 2:
        print("N_CLASSES has to be at least 2. Setting N_CLASSES to 2.")
        return 2
    return n_classes


def validate_residual_learning(residual_learning, zero_padding):
    if residual_learning and not zero_padding:
        print("RESIDUAL_LEARNING requires the matrix shapes not to change. Therefore ZERO_PADDING will be enabled.")
        return True, True
    return residual_learning, zero_padding
