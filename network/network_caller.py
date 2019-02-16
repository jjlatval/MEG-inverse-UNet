from __future__ import division, print_function, unicode_literals
from future.builtins import object
import numpy as np
from os.path import join
from config import N_EPOCHS, WEIGHT_PATH, BATCH_SIZE, SPACING, OPTIMIZER, COST_FUNCTION, \
    CLASS_WEIGHTS, OPTIMIZER_KWARGS, N_CLASSES, RESIDUAL_LEARNING, ZERO_PADDING, REGULARIZATION_KWARGS, NN_TESTS_DIR, \
    CHANNELS, EMPTY_SIGNAL, TRAINING_DATA_PROCESSING, N_DIPOLES
from data_provider.data_provider import GeneratorDataProvider
from network.get_class_weights import approximate_class_frequencies, inverse_frequency_balancing
from network.unet import unet
from network.unet import unet_utils
from parameters.get_network_hyperparams import get_hyperparams
from utils.network_tester import NetworkTester
import six


class NetworkCaller(object):
    def __init__(self, hemisphere, output_path=WEIGHT_PATH, spacing=SPACING, epochs=N_EPOCHS,
                 batch_size=BATCH_SIZE, verification_batch_size=BATCH_SIZE,
                 optimizer=OPTIMIZER, cost=COST_FUNCTION, n_classes=N_CLASSES,
                 class_weights=CLASS_WEIGHTS, residual_learning=RESIDUAL_LEARNING,
                 zero_padding=ZERO_PADDING, optimizer_kwargs=OPTIMIZER_KWARGS,
                 regularization_kwargs=REGULARIZATION_KWARGS, tests_save_dir=NN_TESTS_DIR, channels=CHANNELS,
                 empty_signal=EMPTY_SIGNAL, training_processing=TRAINING_DATA_PROCESSING, n_dipoles=N_DIPOLES):
        """
        NetworkCaller object
        """

        self.hemisphere = hemisphere
        self.output_path = output_path
        self.spacing = spacing
        self.epochs = epochs
        self.batch_size = batch_size  # TODO
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.n_classes = n_classes
        self.n_dipoles = n_dipoles
        self.weights = None
        if class_weights:
            if len(class_weights) == self.n_classes:
                self.class_weights = class_weights
            else:
                print("The length of class weights is different than the number of classes. "
                      "Approximating new weights...")
                self.class_weights = approximate_class_frequencies().values()
        else:
            print("No class weight lists were found. Approximating class weights...")
            self.class_weights = approximate_class_frequencies().values()
        self.class_weights = inverse_frequency_balancing(self.class_weights)
        self.residual_learning = residual_learning
        self.zero_padding = zero_padding
        self.regularization_kwargs = regularization_kwargs
        self.dropout = self.regularization_kwargs.pop("dropout", 1.0)
        self.tests_save_dir = tests_save_dir
        # TODO: more balancing options.
        # self.class_weights = median_frequency_balancing(self.class_weights)
        self.cost = cost
        self.empty_signal = empty_signal
        self.training_processing = training_processing
        self.regularization_kwargs["class_weights"] = self.class_weights
        self.hyperparams = get_hyperparams(self.spacing)

        # TODO: check if whole time series is needed as a verification dataset
        self.verification_batch_size = verification_batch_size
        self.data_provider = GeneratorDataProvider("train", self.hemisphere)

        self.n_channels = self.data_provider.n_channels
        self.channels = channels
        if self.data_provider.location_channel:
            self.n_channels += 3

        self.net = unet.Unet(channels=self.n_channels,
                             n_class=self.data_provider.n_classes,
                             cost=self.cost, regularization_kwargs=self.regularization_kwargs,
                             residual_learning=self.residual_learning,
                             zero_padding=self.zero_padding, **self.hyperparams)

        self.training_iters = int(self.data_provider.t_steps * (1 + self.empty_signal) / self.batch_size)
        self.validation_provider = GeneratorDataProvider("valid", self.hemisphere)
        self.trainer = unet.Trainer(self.net, optimizer=self.optimizer, opt_kwargs=self.optimizer_kwargs)
        self.test_provider = GeneratorDataProvider("test", self.hemisphere)

        self.training_processing = dict.fromkeys(self.training_processing, False)
        self.test_provider_no_processing = GeneratorDataProvider("test", self.hemisphere,
                                                                 training_processing=self.training_processing)
        self.network_tester = NetworkTester()

    def train_network(self):
        print("Training network with %s epochs, %s iters per epoch, %s classes and %s channels with p(dropout)=%s" %
              (self.epochs, self.training_iters, self.data_provider.n_classes, self.data_provider.n_channels,
               self.dropout))
        print("Inputs: [%s], ground truth: %s" % (', '.join(self.data_provider.channels),
                                                  self.data_provider.ground_truth))
        self.weights = self.trainer.train(self.data_provider, self.validation_provider, self.output_path,
                                          training_iters=self.training_iters, epochs=self.epochs, dropout=self.dropout)

    def resume_training(self):
        print("Resuming network training")
        self.weights = self.trainer.train(self.data_provider, self.validation_provider, self.output_path,
                                          training_iters=self.training_iters, epochs=self.epochs,
                                          dropout=self.dropout, restore=True)

    def predict(self, weights=None, n_tests=10, delimiter=str(",")):
        weights = weights or self.weights
        if not weights:
            print("Path to trained weights were not found. Training network...")
            self.train_network()

        final_results = np.array([])

        for t in range(0, n_tests):
            test_x, test_y = six.next(self.test_provider)
            test_x2, test_y2 = six.next(self.test_provider_no_processing)  # No processing used

            prediction = self.net.predict(weights, test_x)
            print("Testing error rate: {:.2f}%".format(
                unet.error_rate(prediction, unet_utils.crop_to_shape(test_y, prediction.shape))))

            img = unet_utils.combine_img_prediction(test_x2, test_y2, prediction)
            unet_utils.save_image(img, join(self.tests_save_dir, "prediction_%s.jpg" % t))

            predicted, dle, sd, oa = self.network_tester.resolution_metrics(test_x2, test_y2, prediction,
                                                                            border_pixels=2)
            results = np.hstack((predicted, dle, sd, oa))

            if final_results.shape[0] == 0:
                final_results = results
            else:
                final_results = np.vstack((final_results, results))

        channels = self.channels[:]
        channels.append("nn")

        header = ["GT_vertex"]
        for r in ["vertex", "DLE", "SD", "OA"]:
            for c in channels:
                header.append(c + "_" + r)

        header = delimiter.join(header)
        if self.n_dipoles == 1:
            np.savetxt(join(self.tests_save_dir, "results.csv"), final_results, fmt=str("%8f"), header=header,
                       delimiter=delimiter)
        else:
            np.savetxt(join(self.tests_save_dir, "results.csv"), final_results, fmt=str("%s"), header=header,
                       delimiter=delimiter)
