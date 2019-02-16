# This file contains configuration parameters that are used throughout the code base
# NOTE: allocate compuating resources in computing_resources.py

from __future__ import unicode_literals, print_function, division


def get_simulation_model_kwargs(n_dipoles, snr):
    """
    Returns simulation model kwargs.
    :param n_dipoles: int
        Number of dipoles to be simulated.
    :param snr: float
        Signal to noise ratio of simulated data.
    :return:
    """
    """
    Explanation of simulation_model_kwargs:
    :param l_freq: 5.0 | float
        Lowest frequency to be included in the simulated raw data.
    :param h_freq: 40.0 | float
        Highest frequency to be included in the simulated raw data. Defaults to 40.0.
    :param phase: str
        Phase of the filter, only used if method='fir'. By default, a symmetric linear-phase FIR filter is constructed.
        If phase='zero' (default), the delay of this filter is compensated for. If phase=='zero-double',
        then this filter is applied twice, once forward, and once backward. If 'minimum', then a minimum-phase,
        causal filter will be used.
    :param fir-window: str
        The window to use in FIR design, can be 'hamming' (default), 'hann' (default in MNE Python 0.13), or 'blackman'.
    :param reject: dict
        Dictionary containing reject tresholds for different sensor types. Defaults to
        {'grad': 4000e-13, 'mag': 4e-12, 'eog': 150e-6}.
    :param channel_types: dict
        Dictionary containing channel types that are included in simulations. Defaults MEG to to True and everything
         else as False. Every dictionary key can have Boolean values besides meg also supporting
         'mag', 'grad', 'planar1' or 'planar2' to select only magnetometers, all gradiometers, or
         a specific type of gradiometer.
    :param cov: instance of Covariance (MNE Python) | str | None
        The sensor covariance matrix used to generate noise. If None, no noise will be added. If 'simple',
         a basic (diagonal) ad-hoc noise covariance will be used. If a string, then the covariance will be loaded.
    :param ecg: Boolean
        Defaults to True
    : param blink: Boolean
        Defaults to True
    :param iir_filter: None | array
        IIR filter coefficients (denominator) e.g. [1, -1, 0.2].
    :param fir_design: firwin | string
        FIR filter design.
    :param n_simulations: int
        Y
    :param n_dipoles: int
        Z
    :param samples_per_dipole: int
        Amount of samples generated per simulated dipole. This is based on Nyquist sampling theorem.
        Has to be higher than 10.
    :param use_cpt True | Boolean
        Use cortical patch statistics for defining normal orientations.
    :param verbose: Boolean
        Enables verbose (default True).
    :param loose: None | float in [0, 1]
        Used for calculating the inverse operator. Value that weights the source variances of the
        dipole components defining the tangent space of the cortical surfaces. Requires surface- based,
        free orientation forward solutions. Defaults to 0.2.
    :param depth: None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed. Defaults to None
    :param lambda2: float
        Regularization parameter used when applying inverse operator. Defaults to 1./9.
    """

    kwargs = {'l_freq': 5.0, 'h_freq': 40.0, 'phase': 'zero', 'fir_window':'hamming',
                         'reject': {'grad': 4000e-13, 'mag': 5e-12, 'eog': 150e-6},
                         'channel_types': {'meg': True, 'eeg': False, 'stim': False, 'eog': False},
                         'cov': 'simple', 'iir_filter': None, 'fir_design': 'firwin', 'ecg': True, 'blink': True,
                         'n_simulations': 3000, 'n_dipoles': n_dipoles, 'samples_per_dipole': 20, 'use_cps': True,
                         'verbose': True, 'loose': 0.2, 'depth': 0.8, 'lambda2': 1. / snr ** 2}
    # Now max freq set to 10Hz
    # Leave samples_per_dipole > 10! TODO: implement frequency range here with Nyquist Shannon Sampling theory
    return kwargs


def get_data_provider_kwargs():
    """
    Get data provider kwargs
    :return: tuple(dict, dict)
    """
    training_data_processing = {'mean_subtraction': False, 'normalize': True, 'binning': False,
                                'onehotencode': False, 'nmf_decomposition': False, 'whiten': False}
    target_data_processing = {'mean_subtraction': False, 'normalize': False, 'binning': True,
                              'onehotencode': True, 'nmf_decomposition': False, 'whiten': False}
    return training_data_processing, target_data_processing


def get_optimizer_kwargs(optimizer):
    """
    Returns a dictionary of optimizer specific kwargs.
    :param optimizer: str
        Either "momentum", "adam", "adagrad", "adadelta" or "rmsprop".
    :return: 
    """
    if optimizer == "momentum":
        kwargs = {"learning_rate": 0.04, "decay_rate": 0.95, "momentum": 0.9}  # Orig moment 0.2
    elif optimizer == "adam":
        kwargs = {"learning_rate": 0.01}
    elif optimizer == "adagrad":
        kwargs = {"learning_rate": 0.1}
    elif optimizer == "adadelta":
        kwargs = {"learning_rate": 1., "rho": 0.95, "epsilon": 1e-6}
    elif optimizer == "rmsprop":
        kwargs = {"learning_rate": 0.1, "decay": 0.95, "momentum": 0.9, "epsilon": 1e-6}
    else:
        raise ValueError("Optimizer %s not understood." % optimizer)
    return kwargs


def get_regularization_kwargs():
    """
    Returns network kwargs in dictionary.
    :return:
    """
    kwargs = {"dropout": 0.8, "batch_normalization": True, "l1_lambda": 0.005,
              "l2_lambda": None, "max_norm": 1e-2, "label_smoothing": 0}
    return kwargs
