# NOTE: running this file requires most likely an Anaconda environment.

from __future__ import unicode_literals
import os
import sip
from config import SUBJECT_NAME
from data_provider.data_provider import DataProvider
import sys

os.environ["ETS_TOOLKIT"] = "qt4"
sip.setapi("QString", 2)
os.environ["QT_API"] = "pyqt"


def plot_dataset(dataset_name, time=None):
    data_model = DataProvider()
    if dataset_name not in data_model.supported_datatypes:
        raise ValueError("%s not in supported datatypes." % dataset_name)
    subject_name = SUBJECT_NAME or 'sample'
    plot_target = getattr(data_model, '%s' % dataset_name)
    plot_target.plot(subject_name)

if __name__ == '__main__':
    plot_dataset(sys.argv[1], sys.argv[2])