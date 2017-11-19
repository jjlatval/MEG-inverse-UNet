# NetworkTester tests the network predictions with some resolution metrics

from __future__ import print_function, division, unicode_literals
from future.builtins import object
import math
import numpy as np
from config import SIMULATION_MODEL_CONFIG, HEMISPHERE, CHANNELS, N_DIPOLES
import mne
import re
from ConfigParser import SafeConfigParser


class NetworkTester(object):
    def __init__(self, config_path=SIMULATION_MODEL_CONFIG, hemisphere=HEMISPHERE, channels=CHANNELS,
                 n_dipoles=N_DIPOLES):

        self.config = SafeConfigParser()
        self.config.read(config_path)

        if not self.config:
            raise IOError("Simulation model config file not found. Simulate data first.")

        for value in self.config.items('simulation_model'):  # Initialize kwargs into DataProvider object
            setattr(self, value[0], value[1])

        self.hemishpere = hemisphere

        # SRC
        self.src = mne.read_source_spaces(self.src_path)

        # Vertices that are in use:
        self.lh_inuse = self.src[0]["inuse"]
        self.rh_inuse = self.src[1]["inuse"]
        self.lh_coordinates = self.src[0]["rr"]  # "nn" is used for directions
        self.rh_coordinates = self.src[1]["rr"]
        self.lh_verts = np.where(self.lh_inuse == 1)[0]
        self.rh_verts = np.where(self.rh_inuse == 1)[0]
        self.n_lh_verts = self.lh_inuse.sum()
        self.n_rh_verts = self.rh_inuse.sum()
        self.lh_xyz = np.zeros([self.lh_inuse.sum(), 3])
        self.rh_xyz = np.zeros([self.lh_inuse.sum(), 3])
        self.__get_used_verts_xyz_coordinates()
        self.euclidean_distance_matrix = np.array([])

        self.channels = channels
        self.n_channels = len(self.channels)
        self.n_dipoles = n_dipoles

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

    def __initialize_euclidean_distance_matrix(self, array1):
        distances = np.empty((array1.shape[1], array1.shape[1]))
        for x in range(0, array1.shape[1]):
            for y in range(x, array1.shape[1]):
                d = self.get_euclidean_distance([x], [y])
                distances[x, y] = d
                distances[y, x] = d
        self.euclidean_distance_matrix = distances

    def get_euclidean_distance(self, vert1, vert2):

        if isinstance(vert1, int):
            vert1 = [vert1]

        if isinstance(vert2, int):
            vert2 = [vert2]

        if len(vert1) != len(vert2):
            raise ValueError("The number of picked maxima has to match with the number of ground truth vertices.")

        if self.hemishpere == "rh":
            coords = self.rh_xyz
        elif self.hemishpere == "lh":
            coords = self.lh_xyz
        else:
            raise ValueError("Hemisphere %s not understood." % self.hemishpere)

        coords1, coords2 = [], []

        for v1, v2 in zip(vert1, vert2):
            coords1.append(coords[v1])
            coords2.append(coords[v2])

        distances = []

        for c1 in coords1:
            distance = []
            for c2 in coords2:
                distance.append(np.linalg.norm(c1 - c2))
            min_index = distance.index(min(distance))
            distances.append(distance[min_index])
            del coords2[min_index]

        return sum(distances) / float(len(vert2))

    def resolution_metrics(self, array1, ground_truth, array2, border_pixels=1, zero_index=0):
        """
        Returns resolution metrics for network predictions
        :param array1: numpy array of shape [n, nx, ny, channels]
        :param ground_truth: numpy array of shape [n, nx, ny, classes]
        :param array2: numpy array of shape [n, nx, ny, classes]
        :param border_pixels: int
            Amount of empty pixels surrounding array2 in order to make its nx, ny equal to array1
        :return dle_array1, dle_array2
        """
        # Remove borders
        border_x = array1.shape[1] - array2.shape[1]
        border_y = array1.shape[2] - array2.shape[2]

        if border_x % 2 == 0:
            x = border_x // 2
            array1 = array1[:, x:-x, :, :]
            ground_truth = ground_truth[:, x:-x, :, :]
        else:
            x1 = border_x // 2 + 1
            x2 = x1 - 1
            array1 = array1[:, x1:-x2, :, :]
            ground_truth = ground_truth[:, x1:-x2, :, :]

        if border_y % 2 == 0:
            y = border_y // 2
            array1 = array1[:, :, y:-y, :]
            ground_truth = ground_truth[:, :, y:-y, :]

        else:
            y1 = border_y // 2 + 1
            y2 = y1 - 1
            array1 = array1[:, :, y1:-y2, :]
            ground_truth = ground_truth[:, :, y1:-y2, :]

        channels = array1.shape[-1]
        classes = array2.shape[-1]

        # Column-normalize matrices:
        array1_psf, array2_psf = array1.copy(), array2.copy()
        for c in range(0, channels):
            array1_psf[0, :, :, c] /= np.max(np.abs(array1_psf[0, :, :, c]), axis=0)

        for c in range(0, classes):
            array2_psf[0, :, :, c] /= np.max(np.abs(array2_psf[0, :, :, c]), axis=0)

        if self.euclidean_distance_matrix.size == 0:
            self.__initialize_euclidean_distance_matrix(array1)

        dle = np.zeros([ground_truth.shape[2], channels + 1])  # including the prediction
        sd = np.zeros([ground_truth.shape[2], channels + 1])
        oa = np.zeros([ground_truth.shape[2], channels + 1])
        if self.n_dipoles == 1:
            predicted_vertices = np.zeros((ground_truth.shape[2], channels + 2))
        else:
            # TODO: implement a more reasonable way of storing multiple results than chararray
            predicted_vertices = np.chararray((ground_truth.shape[2], channels + 2), itemsize=(self.n_dipoles * 8))

        for y in range(0, ground_truth.shape[2]):
            if y >= ground_truth.shape[2] - 1:
                break

            activation_locations = np.unique(np.where(ground_truth[0, :, y, zero_index] != 1.0)[0])  # it is a tuple with a value of e.g. (array([407, 407]), array([0, 1]))
            active_dipoles = len(activation_locations)
            if len(activation_locations) == 0:
                continue
            if self.n_dipoles == 1:
                predicted_vertices[y, 0] = activation_locations
            else:
                predicted_vertices[y, 0] = str(str("|").join(map(str, activation_locations)))

            for c in range(0, channels):
                if active_dipoles > 1:
                    arr1_max = np.argpartition(array1[0, :, y, c], -active_dipoles)[-active_dipoles:]
                    predicted_vertices[y, c + 1] = str(str("|").join(map(str, arr1_max)))
                else:
                    arr1_max = np.argmax(array1[0, :, y, c])
                    if self.n_dipoles > 1:
                        predicted_vertices[y, c + 1] = str(arr1_max)
                    else:
                        predicted_vertices[y, c + 1] = arr1_max

                distances = self.euclidean_distance_matrix[np.argmax(array1[0, :, y, c])]
                psfs = array1_psf[0, :, y, c]
                psfs2 = psfs**2
                final_sd = math.sqrt(np.sum([np.multiply(distances, psfs2)]) / np.sum([psfs2]))

                peak_distance = self.get_euclidean_distance(activation_locations, arr1_max)
                dle[y, c] = peak_distance
                sd[y, c] = final_sd
                oa[y, c] = np.sum([np.absolute(array1_psf[0, :, y, c])])

            if active_dipoles > 1:
                arr2_max = np.argpartition(array2[0, :, y, 1], -active_dipoles)[-active_dipoles:]
                predicted_vertices[y, -1] = str(str("|").join(map(str, arr2_max)))
            else:
                arr2_max = np.argmax(array2[0, :, y, 1])
                if self.n_dipoles > 1:
                    predicted_vertices[y, -1] = str(arr1_max)
                else:
                    predicted_vertices[y, -1] = arr2_max

            distances = self.euclidean_distance_matrix[np.argmax(array2[0, :, y, 1])]
            psfs = array2_psf[0, :, y, 1]
            psfs2 = psfs ** 2
            final_sd = math.sqrt(np.sum([np.multiply(distances, psfs2)]) / np.sum([psfs2]))

            peak_distance = self.get_euclidean_distance(activation_locations, arr2_max)
            dle[y, channels] = peak_distance
            sd[y, channels] = final_sd
            oa[y, channels] = np.sum([np.absolute(array2_psf[0, :, y, 1])])

        delete_indices = []

        if self.n_dipoles > 1:
            for v in range(0, len(predicted_vertices)):
                if re.compile('([1-9])\d|([1-9|])').match(predicted_vertices[v, 0]) is None:
                    delete_indices.append(v)
        else:
            for v in range(0, len(predicted_vertices)):
                if predicted_vertices[v, 0] == 0:
                    delete_indices.append(v)

        dle = np.delete(dle, delete_indices, axis=0)
        sd = np.delete(sd, delete_indices, axis=0)
        oa = np.delete(oa, delete_indices, axis=0)
        predicted_vertices = np.delete(predicted_vertices, delete_indices, axis=0)

        return predicted_vertices, dle, sd, oa




