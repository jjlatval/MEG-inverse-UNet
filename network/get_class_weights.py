from __future__ import unicode_literals, division

import numpy as np
import six
import statistics
from tqdm import tqdm

from config import N_CLASSES, TARGET_DATA_PROCESSING
from data_provider.data_provider import GeneratorDataProvider


def approximate_class_counts(n_classes=N_CLASSES, sample_ts=10, dataset='train', hemisphere='lh'):
    """
    Approximates the class frequency distribution based on a small sample of the dataset.
    :param n_classes: integer
        Number of classes.
    :param sample_ts: integer
        Number of first timesteps used for approximation.
    :param dataset: string
        'train', 'valid' or 'test'.
    :param hemisphere: string
        'lh' or 'rh'.
    :return: dictionary, integer
        Dictionary where class is key and number of occurences is value. 2nd return
    """

    target_data_processing = TARGET_DATA_PROCESSING
    target_data_processing['onehotencode'] = False

    generator = GeneratorDataProvider(dataset, hemisphere, target_processing=target_data_processing)
    class_counts = dict.fromkeys(range(0, n_classes), 0)
    ts = min(sample_ts, generator.t_steps)  # a limited amount of t steps is enough

    print("Approximating class counts...")
    for t in tqdm(range(0, ts)):  # A small sample of a length
        data, target = six.next(generator)
        unique, counts = np.unique(target, return_counts=True)

        for val in zip(unique, counts):
            class_counts[val[0]] += val[1]

    print("Occurences of distinct classes in GeneratorDataProvider.%s_stc for hemisphere: %s in first %s timesteps" % (dataset, hemisphere, sample_ts))
    print(class_counts)

    return class_counts


def approximate_class_frequencies(n_classes=N_CLASSES, sample_ts=10, dataset='train', hemisphere='lh'):
    """ Same as approximate_class_counts but returns frequencies instead. """
    f = approximate_class_counts(n_classes, sample_ts, dataset, hemisphere)
    total = sum(f.values())
    for key, value in f.items():
        f[key] = value / total
    return f


def median_frequency_balancing(weight_freq_list):
    """
    Median frequency balancing for list of weight frequencies.
    :param weight_freq_list: list
        List of weight frequencies.
    :return: list
        List of median frequency weights per class.
    """
    med = statistics.median(weight_freq_list)
    return [med/x for x in weight_freq_list]


def inverse_frequency_balancing(weight_freq_list):
    return [1 / x for x in weight_freq_list]
