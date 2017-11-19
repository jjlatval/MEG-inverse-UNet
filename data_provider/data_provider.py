# data_provider has instances for streaming and processing simulated data into the neural network.

from __future__ import print_function, division, unicode_literals
from future.utils import implements_iterator
from future.builtins import object
import numpy as np
from config import N_CLASSES, SIMULATION_SAVE_PATH, SIMULATION_MODEL_CONFIG, CHANNELS, N_CHANNELS, GROUND_TRUTH, \
    TRAINING_DATA_PROCESSING, TARGET_DATA_PROCESSING, BATCH_SIZE, NETWORK_TYPE, SNR, SPACING, LOCATION_CHANNEL
import mne
from os.path import join
from ConfigParser import SafeConfigParser
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import NMF
import six
from utils.general_utils import most_square_shape
from scipy.cluster.vq import whiten
from scipy import linalg


class DataProvider(object):
    def __init__(self, data_path=SIMULATION_SAVE_PATH, config_path=SIMULATION_MODEL_CONFIG,
                 channels=CHANNELS, n_channels=N_CHANNELS, ground_truth=GROUND_TRUTH,
                 n_classes=N_CLASSES, training_processing=TRAINING_DATA_PROCESSING,
                 target_processing=TARGET_DATA_PROCESSING, dataset_type='train', a_min=0, a_max=1, snr=SNR):

        # TODO: Add distinct zero class index
        self.config = SafeConfigParser()
        self.config.read(config_path)

        if not self.config:
            raise IOError("Simulation model config file not found. Simulate data first.")

        for value in self.config.items('simulation_model'):  # Initialize kwargs into DataProvider object
            setattr(self, value[0], value[1])

        # Supported data types
        self.supported_datatypes = {'stc', 'mne', 'raw', 'sloreta', 'dspm'}
        self.available_hemispheres = {'lh', 'rh'}
        self.dataset_types = {'train':0, 'valid':1, 'test':2}
        if dataset_type not in self.dataset_types.keys():
            raise ValueError("Dataset type %s not supported." % dataset_type)
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.channels = channels
        self.n_channels = n_channels
        self.ground_truth = ground_truth
        self.n_classes = n_classes
        self.anomaly_detection = self.n_classes == 2  # Anomaly detection mode on if N_CLASSES = 2

        # Set processing steps for train and target data
        self.train_processing = training_processing
        self.target_processing = target_processing
        self.snr = snr

        # self.__validate_data_provider_call()

        data_name = 'simulated-data-%s' % self.dataset_types[self.dataset_type]
        self.raw = mne.io.read_raw_fif(join(self.data_path, data_name + '_raw.fif'), preload=True)
        self.stc = mne.read_source_estimate(join(self.data_path, data_name))
        self.mne = mne.read_source_estimate(join(self.data_path, data_name + '_mne'))
        self.sloreta = mne.read_source_estimate(join(self.data_path, data_name + '_sloreta'))
        self.dspm = mne.read_source_estimate(join(self.data_path, data_name + '_dspm'))

        # Limits for normalization:
        self.a_min = a_min
        self.a_max = a_max
        self.enc = OneHotEncoder(n_values=self.n_classes)

        self.raw_processed, self.stc_processed, self.mne_processed, self.sloreta_processed, self.dspm_processed =\
            self.raw.copy(), self.stc.copy(), self.mne.copy(), self.sloreta.copy(), self.dspm.copy()
        self.raw_bins = self.stc_bins = self.mne_bins = self.sloreta_bins = self.dspm_bins = None
        self.__preprocess_datasets()

        self.stc_binned = self.stc_processed.copy()
        self.mne_binned = self.mne_processed.copy()
        self.sloreta_binned = self.sloreta_processed.copy()
        self.dspm_binned = self.dspm_processed.copy()
        self.zero_bin = 0
        self.__bin_datasets()
        self.__get_zero_bin()

        # SRC
        self.src = mne.read_source_spaces(self.src_path)

        # Vertices that are in use:
        self.lh_coordinates = self.src[0]["rr"]  # "nn" is used for directions
        self.rh_coordinates = self.src[1]["rr"]
        self.lh_inuse = self.src[0]["inuse"]
        self.rh_inuse = self.src[1]["inuse"]
        self.lh_verts = np.where(self.lh_inuse == 1)[0]
        self.rh_verts = np.where(self.rh_inuse == 1)[0]
        self.n_lh_verts = self.lh_inuse.sum()
        self.n_rh_verts = self.rh_inuse.sum()
        self.t_steps = self.stc.shape[1]

        self.lh_xyz = np.zeros([self.lh_inuse.sum(), 3])
        self.rh_xyz = np.zeros([self.lh_inuse.sum(), 3])
        self.__get_used_verts_xyz_coordinates()

    def __validate_data_provider_call(self):
        # Test that data shapes match
        # TODO check channels and ground truth here as well
        if not(self.train_stc.shape[1] == self.valid_stc.shape[1] == self.test_stc.shape[1] ==
                   self.train_mne.shape[1] == self.valid_mne.shape[1] == self.test_mne.shape[1]):
            raise ValueError("Time series do not match between all simulated data. Got times: "
                             "self.train_stc: %s, self.valid_stc: %s, self.test_stc: %s, "
                             "self.train_mne: %s, self.valid_mne: %s, self.test_mne: %s" %
                             (self.train_stc.shape[1], self.valid_stc.shape[1], self.test_stc.shape[1],
                             self.train_mne.shape[1], self.valid_mne.shape[1], self.test_mne.shape[1]))
        if not self.spacing == SPACING:
            raise ValueError("Spacing in config.py and config.ini are different. "
                             "Did you change it after you simulated data?")

    def __get_used_verts_xyz_coordinates(self):
        i = 0
        for coords in range(0, self.lh_coordinates.shape[0]):
            if self.lh_inuse[coords] == 1:
                self.lh_xyz[i] = self.lh_coordinates[coords]
                i += 1

        i = 0
        for coords in range(0, self.rh_coordinates.shape[0]):
            if self.rh_inuse[coords] == 1:
                self.rh_xyz[i] = self.rh_coordinates[coords]
                i += 1

    @staticmethod
    def filter_low_values(data, limit=0.35, fill_val=0.0):
        np.putmask(data, abs(data) < np.amax(data) * limit, fill_val)
        return data

    @staticmethod
    def clip_outliers(data, m=3):
        mean = np.mean(data)
        std = np.std(data)
        min_val = mean - m * std
        max_val = mean + m * std
        return np.clip(data, min_val, max_val)

    @staticmethod
    def nmf_decomposition(data):
        model = NMF(n_components=2, init='random', random_state=0)
        W = model.fit_transform(data)
        H = model.components_
        return np.dot(W, H)

    @staticmethod
    def mean_subtraction(data):
        """
        Subtracts means of a 2D matrix by each axis
        :param data: 2D numpy array
        :return: data: 2D numpy array with means subtracted by axis
        """
        if len(data.shape) != 2:
            raise ValueError("Mean subtraction only supports 2D matrices.")
        col_means = np.mean(data, axis=0)
        row_means = np.mean(data, axis=1)
        data = np.add(data, -col_means[np.newaxis, :])
        data = np.add(data, -row_means[:, np.newaxis])
        return data

    @staticmethod
    def normalize_data(data):
        # Normalize between [0, 1]
        return data / data.max(axis=0)

    @staticmethod
    def whiten_data(data):
        return whiten(data)

    @staticmethod
    def svd(data):
        observations, features = data.shape
        U, s, V = linalg.svd(data)
        S = linalg.diagsvd(s, observations, features)
        return S

    def bin_data(self, data, bins):
        if self.n_classes % 2 == 0:
            data = np.absolute(data)
        data = np.digitize(data, bins)
        return data

    def __preprocess_datasets(self):

        print("Preprocessing datasets...")
        for c in self.channels:
            target = getattr(self, c + '_processed')
            # target._data = self.clip_outliers(target.data)
            if self.train_processing["mean_subtraction"] is True:
                target._data = self.mean_subtraction(target.data)
                print("Mean subtraction for %s done." % c)
            if self.train_processing["normalize"] is True:
                target._data = self.normalize_data(target.data)
                print("Normalization for %s done." % c)
            if self.train_processing["whiten"] is True:
                target._data = self.whiten_data(target.data)
                print("Whitening for %s done." % c)
            if self.train_processing["nmf_decomposition"] is True:
                target._data = self.nmf_decomposition(target.data)
                print("NMF decomposition for %s done." % c)
            # target._data = self.svd(target.data)

        for g in self.ground_truth:
            target = getattr(self, g + '_processed')
            if self.target_processing["mean_subtraction"] is True:
                target._data = self.mean_subtraction(target.data)
                print("Mean subtraction for %s done." % g)
            if self.target_processing["mean_subtraction"] is True:
                target._data = self.normalize_data(target.data)
                print("Normalization for %s done." % g)
            if self.target_processing["whiten"] is True:
                target._data = self.whiten_data(target.data)
                print("Whitening for %s done." % g)
            if self.target_processing["nmf_decomposition"] is True:
                target._data = self.nmf_decomposition(target.data)
                print("NMF decomposition for %s done." % g)

    def __find_bins(self, data):

        def get_bins(maxv, snr, n_class, anomaly_detection=False):
            # median = np.median(data)
            # threshold = median * 2 / (snr + 1)
            threshold = 10e-20
            pos_bins = np.linspace(threshold, maxv, n_class // 2)
            neg_bins = -1 * np.flip(pos_bins, axis=0)
            if anomaly_detection:
                bins = pos_bins
            else:
                bins = np.append(neg_bins, pos_bins)
            # bins = np.array([-threshold, threshold])
            # bins = np.linspace(threshold, maxv, n_class - 1)
            return bins

        target_bins = get_bins(np.amax(data), self.snr, self.n_classes, anomaly_detection=self.anomaly_detection)

        return target_bins

    def __get_zero_bin(self):
        # TODO: this uses only STC now
        self.zero_bin = np.digitize(0, self.stc_bins, right=True).tolist()

    def __bin_datasets(self):

        for c in self.channels:
            target = getattr(self, c + '_binned')
            if self.train_processing["binning"] is True:
                bins = self.__find_bins(target.data)
                if c == 'mne':  # TODO: for some reason setattr did not work.
                    self.mne_bins = bins
                elif c == 'sloreta':
                    self.sloreta_bins = bins
                elif c == 'dspm':
                    self.dspm_bins = bins

                target._data = np.digitize(target.data, bins, right=True)
                print("Binning done for %s." % c)

        for g in self.ground_truth:
            target = getattr(self, g + '_binned')
            if self.target_processing["binning"] is True:
                bins = self.__find_bins(target.data)
                if g == 'stc':
                    self.stc_bins = bins
                if self.anomaly_detection:
                    target._data = np.absolute(target.data)
                target._data = np.digitize(target.data, bins, right=True)
                print("Binning done for %s." % g)

    def _init_vector_with_min_val(self, length):
        return np.full(length, self.a_min).astype(np.float)

    def _init_vector_with_zero_bin(self, length):
        return np.full(length, self.zero_bin).astype(np.float)

    def _get_source_estimate_vector(self, se_instance, t_step, d_type, hemisphere):
        """
        Returns a vector of a given source estimate instance at a given timestep and hemisphere.
        :param se_instance: mne.SourceEstimate
            A source estimate of type 'mne' or 'stc'.
        :param t_step: int
            Desired timestep index in integer format
        :param d_type: str: 'stc' | 'mne'
            Desired SourceEstimate type indicator
        :param hemisphere: str: 'lh' | 'rh'
            Target hemisphere.
        :return: numpy array
            Vector of given hemisphere for given time step t.
        """
        if d_type in ['stc', 'mne', 'sloreta', 'dspm']:  #TODO
            if hemisphere == 'lh':
                all_verts = self.lh_verts
                data_verts = se_instance.lh_vertno
                data = se_instance.lh_data
            elif hemisphere == 'rh':
                all_verts = self.rh_verts
                data_verts = se_instance.rh_vertno
                data = se_instance.rh_data
            else:
                raise ValueError("Hemisphere has to be 'lh' or 'rh'. Found %s." % hemisphere)
            if d_type == 'stc':
                vector = self._init_vector_with_zero_bin(len(all_verts))
            else:
                vector = self._init_vector_with_min_val(len(all_verts))
            for v in range(0, data_verts.shape[0]):
                vector[np.where(data_verts[v] == all_verts)[0][0]] = data[v][t_step]
        else:
            raise ValueError("Datatype of %s not supported. Supported datatypes are 'stc' and 'mne'." % d_type)

        return vector

    @staticmethod
    def _get_raw_vector(data, t_step):
        # TODO: Prepare raw vector in such a way that it can in interpolated to match stc and mne
        return data[:, t_step][0]


@implements_iterator
class GeneratorDataProvider(DataProvider):
    """ GeneratorDataProvider provides the data within DataProvider as a generator """
    def __init__(self, dataset_type, hemisphere, data_path=SIMULATION_SAVE_PATH, a_min=0, a_max=1,
                 config_path=SIMULATION_MODEL_CONFIG, batch_size=BATCH_SIZE, n_classes=N_CLASSES, channels=CHANNELS,
                 n_channels=N_CHANNELS, location_channel=LOCATION_CHANNEL,
                 ground_truth=GROUND_TRUTH, training_processing=TRAINING_DATA_PROCESSING,
                 target_processing=TARGET_DATA_PROCESSING, network_type=NETWORK_TYPE):

        super(GeneratorDataProvider, self).__init__(data_path, config_path, channels, n_channels, ground_truth,
                                                    n_classes, training_processing, target_processing,
                                                    dataset_type, a_min, a_max)
        self.location_channel = location_channel
        self.hemisphere = hemisphere
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.network_type = network_type

        # Set indices for iterating over the dataset
        self.current_batch_idx = 0
        self.n_vertices = None  # This value is set in __validate_generator_call()
        self.__validate_generator_call()

        if self.network_type == 'unet' and self.batch_size == 1:
            self.target_shape = most_square_shape(self.n_vertices)
        else:
            self.target_shape = (self.n_vertices, self.batch_size)

        if self.location_channel:
            self.location_template = self.__init_location_channel_template()
        else:
            self.location_template = None
        # self.target_shape = most_square_shape(self.n_vertices)

        # Generator
        self.generators = self.get_generators()
        print("GeneratorDataProvider for dataset %s loaded successfully." % self.dataset_type)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data, labels = six.next(self.generators)
        except StopIteration:
            self.refresh_generators()  # Refresh generators
            data, labels = six.next(self.generators)

        # data = train_data.reshape(1, self.target_shape[1], 376, self.n_channels)

        if self.network_type == 'lstm':
            data = data.reshape(self.target_shape[0], self.target_shape[1], self.n_channels)

            if self.target_processing['onehotencode'] is True:
                labels = self.onehotencode(labels)
                labels = labels.reshape(self.target_shape[0], self.target_shape[1], self.n_classes)
            else:
                labels = labels.reshape(self.target_shape[0], self.target_shape[1], 1)


        else:

            data = data.reshape(1, self.target_shape[0], self.target_shape[1], self.n_channels)

            if self.location_channel:
                empty_frame = self.location_template.copy()
                empty_frame[:, :, :, :self.n_channels] = data
                data = empty_frame

            if self.target_processing['onehotencode'] is True:
                labels = self.onehotencode(labels)
                labels = labels.reshape(1, self.target_shape[0], self.target_shape[1], self.n_classes)

            else:
                labels = labels.reshape(1, self.target_shape[0], self.target_shape[1], 1)

        return data, labels

    def onehotencode(self, labels):
        self.enc.fit(labels)
        return self.enc.fit_transform(labels).toarray()

    def __init_location_channel_template(self):

        if self.hemisphere == 'rh':
            x, y, z = self.rh_xyz[:, 0], self.rh_xyz[:, 1], self.rh_xyz[:, 2]
        elif self.hemisphere == 'lh':
            x, y, z = self.lh_xyz[:, 0], self.lh_xyz[:, 1], self.lh_xyz[:, 2]
        else:
            raise ValueError("Hemisphere %s not understood." % self.hemisphere)

        # verts = np.arange(self.n_vertices)

        if self.train_processing["normalize"] is True:
            x = self.normalize_data(x)
            y = self.normalize_data(y)
            z = self.normalize_data(z)
            # verts = self.normalize_data(verts)
            print("Normalization for location channel vertex indices done.")

        x = np.tile(x, (self.batch_size, 1)).T
        y = np.tile(y, (self.batch_size, 1)).T
        z = np.tile(z, (self.batch_size, 1)).T
        # verts = np.tile(verts, (self.batch_size, 1)).T
        template = np.zeros([1, self.target_shape[0], self.target_shape[1], self.n_channels + 3])
        template[:, :, :, self.n_channels] += x
        template[:, :, :, self.n_channels +1] += y
        template[:, :, :, self.n_channels + 2] += z
        return template

    def __validate_generator_call(self):
        if self.dataset_type not in self.dataset_types.keys():  # train, valid or test
            raise ValueError(
                "Generator type %s not understood. Use 'train', 'valid' or 'test'." % self.dataset_type)
        if self.hemisphere not in self.available_hemispheres:  # lh, rh
            raise ValueError("Hemisphere type %s not in supported hemisphere types." % self.hemisphere)
        if self.hemisphere == 'lh':
            self.n_vertices = self.n_lh_verts
            self.verts_in_use = self.lh_inuse
        else:
            self.n_vertices = self.n_rh_verts
            self.verts_in_use = self.rh_inuse
        self.full_src_indices = np.where(self.verts_in_use == 1)

    def _get_data_vector(self, ch, t):
        """ Returns a vector for a target channel at given timepoint t. """
        data_instance = getattr(self, ch + '_binned')
        if ch == 'raw':
            vector = self._get_raw_vector(data_instance, t)
        else:
            if self.hemisphere in self.available_hemispheres:
                vector = self._get_source_estimate_vector(data_instance, t, ch, self.hemisphere)
            else:
                raise ValueError("Hemisphere %s not understood. Supported types: %s" % 
                                 (self.hemisphere, self.available_hemispheres))
        return vector

    def refresh_generators(self):
            self.generators = self.get_generators()

    def get_generators(self):

        i = 0
        train_generators, target_generator = None, None
        while i < self.batch_size:
            if self.t_steps - self.current_batch_idx * self.batch_size < self.batch_size:
                self.current_batch_idx = 0

            start_id = self.current_batch_idx * self.batch_size

            for t in range(start_id, self.t_steps):
                channel_generator = None
                for c in range(0, len(self.channels)):
                    if channel_generator is None:
                        channel_generator = np.array(self._get_data_vector(self.channels[c], t))
                    else:
                        channel_generator = np.column_stack((channel_generator,
                                                             self._get_data_vector(self.channels[c], t)))

                channel_generator = np.expand_dims(channel_generator, axis=1)

                if train_generators is None:
                    train_generators = np.array(channel_generator)
                else:
                    train_generators = np.concatenate((train_generators, channel_generator), axis=1)

                if target_generator is None:
                    target_generator = np.array(self._get_data_vector(self.ground_truth[0], t))
                else:
                    target_generator = np.column_stack((target_generator,
                                                        self._get_data_vector(self.ground_truth[0], t)))
                i += 1
                if i == self.batch_size:
                    self.current_batch_idx += 1
                    break

        yield train_generators, target_generator
