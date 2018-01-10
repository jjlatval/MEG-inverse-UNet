# This file contains configuration parameters

from __future__ import unicode_literals, print_function, division
import os
from os.path import join
from mne.datasets import sample
from parameters.parameters import get_simulation_model_kwargs, get_data_provider_kwargs, get_optimizer_kwargs,\
    get_regularization_kwargs
from validation.validation import validate_cpu_threads, validate_spacing, validate_channels_and_ground_truth,\
    validate_n_classes, validate_residual_learning

#######################
# COMPUTING RESOURCES #
#######################

CPU_THREADS = 8

#########
# PATHS #
#########

ROOT = os.path.abspath(os.path.dirname(__file__))
SIMULATION_MODEL_PATH = join(ROOT, 'simulation_model')
SIMULATION_SAVE_PATH = join(SIMULATION_MODEL_PATH, 'simulated_data')
NETWORK_PATH = join(ROOT, 'network')
WEIGHT_PATH = join(NETWORK_PATH, 'weights')
NETWORK_VALIDATION_PATH = join(NETWORK_PATH, 'validation')
UTILS_PATH = join(ROOT, 'utils')
PREDICTION_PATH = join(ROOT, 'epoch_validation')

##############
# DATA MODEL #
##############

if not os.environ.get("FREESURFER_HOME"):
    print("FREESURFER_HOME environment variable not found. Have you installed Freesurfer properly?")

SPACING = 'ico3'  # Source space grid spacing. Options: ico3, ico4, ico5, oct4,  oct5 or oct6
DATA_PATH = None
#SUBJECTS_DIR = os.environ.get("SUBJECTS_DIR") or join(sample.data_path(), 'subjects')
SUBJECTS_DIR = join(sample.data_path(), 'subjects')
os.environ["SUBJECTS_DIR"] = SUBJECTS_DIR
SUBJECT_NAME = None
RAW_PATH = None
BEM_PATH = None
COV_PATH = None
TRANS_PATH = None
SRC_PATH = None
RAW_EMPTY_ROOM_PATH = None

EMPTY_SIGNAL = 0.5  # Proportion of empty signal in addition to simulated data
SNR = 1.0
N_DIPOLES = 1  # If N_DIPOLES = 1
# then the simulation model will trigger a mode where it simulates each individual

SIMULATION_MODEL_KWARGS = get_simulation_model_kwargs(N_DIPOLES, SNR)
SIMULATION_MODEL_CONFIG = join(SIMULATION_MODEL_PATH, 'config.ini')

#############################
# DATA PROVIDER PARAMETERS #
#############################

TRAINING_DATA_PROCESSING , TARGET_DATA_PROCESSING = get_data_provider_kwargs()

#############################
# NEURAL NETWORK PARAMETERS #
#############################

# Basic network parameters
NETWORK_TYPE = 'unet'  # Supported network types: 'unet', '1dcnn', 'lstm' or 'seq2seq'
# NOTE: Running the network for predicting is only supported in unet at this moment

BATCH_SIZE = 642  # Batch size for neural network training.

N_EPOCHS = 350  # Desired amount of epochs used for neural network training
CHANNELS = ['mne', 'sloreta', 'dspm']  # Supported channel types: 'mne', 'sloreta', 'dspm'.
LOCATION_CHANNEL = False  # Passes the vertex x, y and z coordinates as 3 location channels to the network
GROUND_TRUTH = ['stc']  # Only one ground truth available at a time
HEMISPHERE = 'rh'  # Available options: 'lh', 'rh, or 'both'
N_CLASSES = 2  # The number of classes for convolutional network
# NOTE: if N_CLASSES==2 then absolute values of ground truth will be used
# CLASS_WEIGHTS = [0.998, 0.112]
CLASS_WEIGHTS = [0.99925, 0.00075] # [0.00077803738317757005, 0.99844236760124616, 0.00077959501557632398]  # You can pre-calculate these by
# running class_imbalance.py

# Neural network activation, cost and gradient optimizers.
ACTIVATION_TYPE = 'relu'  # Available ptions: 'sigmoid', 'tanh', 'softmax', 'relu'. Will default to 'relu'
OUTPUT_ACTIVATION_TYPE = 'softmax'  # Softmax is better at the final classification task than relu.
COST_FUNCTION = "cross_entropy"
OPTIMIZER = "momentum"  # The optimization method used for gradient descent.
# Available methods: "momentum", "adam", "adagrad", "adadelta" or "rmsprop".
"""
Notes about certain optimizers:
momentum seems to be the best optimizer for actually finding a solution.
momentum, adam and rmsprop need to have a 'suitable' learning_rate chosen - otherwise the gradient may explode or vanish
adadelta > adagrad
"""
OPTIMIZER_KWARGS = get_optimizer_kwargs(OPTIMIZER)

# Regularization methods
REGULARIZATION_KWARGS = get_regularization_kwargs()


ZERO_PADDING = True  # TODO for some reason zero padding does not work if layer size < 5
RESIDUAL_LEARNING = False  # Adds the ground truth to the final weight output of the network for residual learning

# Weight paths
FINAL_WEIGHT_NAME = 'model.cpkt'
FINAL_WEIGHTS = join(WEIGHT_PATH, FINAL_WEIGHT_NAME)

#################################
# NEURAL NETWORK TESTING PARAMS #
#################################

NN_N_TESTS = 100  # TODO: not in use
NN_TESTS_DIR = join(ROOT, 'network_tests')


#########################################
# CONFIG VALIDATION - DO NOT OVERWRITE! #
#########################################

CPU_THREADS = validate_cpu_threads(CPU_THREADS)
SPACING = validate_spacing(SPACING)
CHANNELS, GROUND_TRUTH, N_CHANNELS = validate_channels_and_ground_truth(CHANNELS, GROUND_TRUTH)
N_CLASSES = validate_n_classes(N_CLASSES)
RESIDUAL_LEARNING, ZERO_PADDING = validate_residual_learning(RESIDUAL_LEARNING, ZERO_PADDING)
