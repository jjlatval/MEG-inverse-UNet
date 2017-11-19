from __future__ import unicode_literals
from config import SPACING, BATCH_SIZE


def get_hyperparams(spacing=SPACING, batch_size=BATCH_SIZE):
    if spacing == 'ico3':
        if batch_size <= 10:  # TODO: test!
            hyperparams = {"layers": 1, "filter_size": 4, "features_root": 8, "pool_size": 2}
        elif 10 < batch_size <= 20:
            hyperparams = {"layers": 2, "filter_size": 4, "features_root": 16, "pool_size": 2}
        elif 20 < batch_size <= 30:
            hyperparams = {"layers": 3, "filter_size": 4, "features_root": 32, "pool_size": 2}
        elif 30 < batch_size <= 50:
            hyperparams = {"layers": 4, "filter_size": 4, "features_root": 64, "pool_size": 2}
        elif 50 < batch_size <= 100:
            hyperparams = {"layers": 4, "filter_size": 4, "features_root": 64, "pool_size": 2}
        elif 100 < batch_size <= 200:
            hyperparams = {"layers": 3, "filter_size": 4, "features_root": 64, "pool_size": 2}
        elif 200 < batch_size <= 1000:
            hyperparams = {"layers": 5, "filter_size": 4, "features_root": 64, "pool_size": 2}
        else:
            hyperparams = {"layers": 1, "filter_size": 4, "features_root": 16, "pool_size": 2}
    else:
        raise ValueError("Spacing %s not understood." % spacing)
    return hyperparams
