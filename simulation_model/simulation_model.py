from __future__ import print_function, division, unicode_literals
from future.builtins import object, range
import numpy as np
import mne
import os
from os.path import join
from mne.simulation import simulate_sparse_stc, simulate_raw, simulate_evoked, source
from mne.time_frequency import fit_iir_model_raw
from mne.minimum_norm import apply_inverse_raw
import re
from config import CPU_THREADS, SIMULATION_SAVE_PATH, SPACING, DATA_PATH, SUBJECT_NAME, RAW_PATH, BEM_PATH, \
    COV_PATH, TRANS_PATH, SRC_PATH, SUBJECTS_DIR, RAW_EMPTY_ROOM_PATH, SNR, HEMISPHERE, EMPTY_SIGNAL
from simulation_model_config import create_config
from mne.preprocessing import create_ecg_epochs, create_eog_epochs
from mne.preprocessing import compute_proj_ecg, compute_proj_eog, ICA, run_ica
import random


# TODO: FIX channel_types and rejecting channels


class SimulationModel(object):
    def __init__(self, data_path=DATA_PATH, subjects_dir=SUBJECTS_DIR, subject_name=SUBJECT_NAME,
                 raw_path=RAW_PATH, bem_path=BEM_PATH, cov_path=COV_PATH, trans_path=TRANS_PATH,
                 src_path=SRC_PATH, raw_empty_room_path=RAW_EMPTY_ROOM_PATH, spacing=SPACING,
                 save_folder=SIMULATION_SAVE_PATH, bad_channels=None, hemisphere=HEMISPHERE, empty_signal=EMPTY_SIGNAL,
                 **kwargs):
        # TODO: Add option to compute covariance based on simulated data
        """
        Makes a data model of raw MEG data and enables simulation of new data based on MEG data templates.

        :param data_path: str
            The path where MEG data is. If None, then mri_dir, meg_dir,
            subject_name and meg_file inputs are ignored.
        :param subjects_dir: str
            X
        :param mri_folder: str
            The directory under data_path where MRI data of subjects is.
        :param meg_folder: str
            The directory under data_path where MEG data of subjects is.
        :param subject_name: str
            The name of the subject.
        :param meg_file: str
            The name of the raw data file containing the channel location and types.
        :param bem_file:
        :param trans_file: str
            The transformation file used for co-registration of head and sensors.
        :param spacing: str
            Default source space grid point spacing. Supported types: ico* and oct* Defaults to oct6.
        :param save_folder:
        :return:
        """

        # Initialize some default parameters
        self.l_freq = self.h_freq = self.phase = self.fir_window = self.channel_types = \
            self.verbose = self.reject = self.loose = self.lambda2 = self.iir_filter = self.ecg = \
            self.n_dipoles = self.samples_per_dipole = self.depth = self.blink = self.n_simulations = \
            self.fir_design = self.times_per_dipole = None

        for key in kwargs:  # Initialize kwargs into SimulationModel object
            setattr(self, key, kwargs[key])

        self.spacing = spacing
        if not all([data_path, subjects_dir, subject_name, raw_path, bem_path, cov_path, trans_path, src_path]): # TODO: raw_empty_room
            print('You need to set data_path, subject_name, raw_path, bem_path, cov_path, trans_path and src_path in '
                  'order to use custom subject data.')
            print('Initializing with MNE sample subject data instead...')
            self.subjects_dir = subjects_dir
            self.__init_with_mne_sample_data()
        else:
            self.data_path = data_path
            self.subjects_dir = subjects_dir
            self.subject_name = subject_name
            self.raw_path = raw_path
            self.bem_path = bem_path
            self.cov_path = cov_path
            self.trans_path = trans_path
            self.src_path = src_path
            self.raw_empty_room_path = raw_empty_room_path

        if not spacing or re.compile('([i-o])\w{2}\d').match(spacing) is None:
            raise ValueError("Incorrect spacing geometry. "
                             "Use ico3, ico4, ico5, oct3, oct4, oct5 or oct6 instead.")

        self.hemisphere = hemisphere
        self.empty_signal = empty_signal
        # Load data
        self.orig_data = mne.io.read_raw_fif(self.raw_path, preload=True)  # load raw fif data
        self.orig_data.set_eeg_reference('average', projection=True)
        self.orig_data.apply_proj()  # Apply SSP

        self.bad_channels = ['MEG 2443', 'EEG 053']
        # Mark bad channels
        if self.bad_channels:  # Bad channels have to be in format: ['MEG 2443', 'EEG 053']
            self.orig_data.info['bads'] = self.bad_channels

        self.picks = mne.pick_types(self.orig_data.info, meg=self.channel_types['meg'], eeg=self.channel_types['eeg'],
                                    eog=self.channel_types['eog'], stim=self.channel_types['stim'], ecg=self.ecg,
                                    exclude='bads')
        # Filter data
        self.orig_data.filter(l_freq=self.l_freq, h_freq=self.h_freq, phase=self.phase, fir_window=self.fir_window,
                              fir_design=self.fir_design)

        # Init basic parameters for simulation
        self.src = mne.read_source_spaces(self.src_path)
        self.rh_labels = np.where(self.src[1]["inuse"] == 1)[0]
        self.lh_labels = np.where(self.src[0]["inuse"] == 1)[0]
        self.fwd = self.sim_data = self.sim_stc = None
        self.spacing_t_steps = self.__get_spacing_t_steps()

        # Setup parameters for generating simulated raw data
        self.sim_vertices = np.array([])

        # Inverse solutions
        self.mne = self.dspm = self.sloreta = None

        self.save_path = save_folder
        self.raw_empty_room = mne.io.read_raw_fif(self.raw_empty_room_path, preload=True)
        self.raw_empty_room.info['bads'] = [bb for bb in self.orig_data.info['bads'] if 'EEG' not in bb]
        self.raw_empty_room.add_proj([pp.copy() for pp in self.orig_data.info['projs'] if 'EEG' not in pp['desc']])
        # self.raw_empty_room.set_eeg_reference()
        # self.cov = mne.read_cov(self.cov_path)
        #self.cov = None
        self.cov = mne.compute_raw_covariance(self.raw_empty_room, tmin=0, tmax=None, n_jobs=CPU_THREADS)
        # self.cov = self.compute_covariance(self.orig_data)
        # self.cov = 'simple'
        self.data_template = None

    def __get_spacing_t_steps(self):
        if self.hemisphere == 'rh':
            return self.src[0]["inuse"].sum()
        elif self.hemisphere == 'lh':
            return self.src[1]["inuse"].sum()
        return self.src[0]["inuse"].sum() + self.src[1]["inuse"].sum()

    def __init_with_mne_sample_data(self):
        """
        Initializes MEG, BEM, Source space and tranforms with MNE sample data
        """
        self.data_path = os.path.abspath(join(self.subjects_dir, os.pardir))
        self.subject_name = 'sample'

        self.raw_path = join(self.data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
        self.info = mne.io.read_info(self.raw_path)
        self.bem_path = join(self.data_path, 'subjects', 'sample', 'bem', 'sample-5120-5120-5120-bem-sol.fif')
        src_file = 'sample-' + self.spacing[0:3] + '-' + self.spacing[-1] + '-src.fif'
        src_path = join(self.data_path, 'subjects', 'sample', 'bem', src_file)
        if not os.path.isfile(src_path):
            print("Source space file not found, creating new source space file...")
            mne.setup_source_space(str('sample'), spacing=self.spacing)

        self.src_path = src_path
        # self.trans_path = None
        self.trans_path = join(self.data_path, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')
        self.cov_path = join(self.data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
        self.raw_empty_room_path = join(self.data_path, 'MEG', 'sample', 'ernoise_raw.fif')

    def save_simulated_data(self):
        n_data = 3  # Create 3 files: one for training, one for validating and one for testing
        for i in range(0, n_data):
            f_name_base = 'simulated-data-'
            data, stc = self.simulate_raw_data()
            # TODO: enable all distinct inverse methods!
            self.calculate_forward_solution(data)
            self.mne = self.calculate_inverse_solution(data, method='MNE')
            self.sloreta = self.calculate_inverse_solution(data, method='sLORETA')
            self.dspm = self.calculate_inverse_solution(data, method='dSPM')
            f_name = join(self.save_path, (f_name_base + str(i)))
            data.save(f_name + '_raw.fif', overwrite=True)
            stc.save(f_name)
            self.mne.save(f_name + '_mne')  # This automatically saves the files for both hemispheres
            self.sloreta.save(f_name + '_sloreta')
            self.dspm.save(f_name + '_dspm')

        # Also save a config
        config_dict = {'data_path': self.data_path, 'subjects_dir': self.subjects_dir,
                                        'subject_name': self.subject_name, 'raw_path': self.raw_path,
                                        'bem_path': self.bem_path, 'cov_path': self.cov_path,
                                        'trans_path': self.trans_path, 'src_path': self.src_path,
                       'spacing': self.spacing}
        config_dict = dict((k, v) for k, v in config_dict.iteritems() if v)
        config_params = {'simulation_model': config_dict}
        # TODO cov and trans paths
        create_config(params=config_params)

    #def compute_covariance(self, data, method='shrunk'):  # method='auto' TODO
    #    print("Computing covariance from raw empty room noise...")
    #    return mne.compute_raw_covariance(data, tmin=0, tmax=None, reject=self.reject, flat=None, picks=self.picks, n_jobs=CPU_THREADS)

    def calculate_forward_solution(self, data):
        """
        Calculates the forward solution
        :param data:
        :return:
        """
        print("Calculating forward solution...")
        self.fwd = mne.make_forward_solution(data.info, self.trans_path, self.src, self.bem_path, n_jobs=CPU_THREADS,
                                             verbose=self.verbose)
        self.fwd = mne.pick_types_forward(self.fwd, meg=self.channel_types['meg'], eeg=self.channel_types['eeg'], exclude=data.info['bads'])
        return self.fwd

    def compute_covariance(self, data):
        events = mne.find_events(data)
        epochs = mne.Epochs(data, events, tmin=-0.2, tmax=0.5, proj=True, picks=self.picks, baseline=(None, 0),
                            preload=True, reject=self.reject, add_eeg_ref=True)
        covariance = mne.compute_covariance(epochs)
        return covariance

    def calculate_inverse_solution(self, data, method='dSPM'):
        """
        Calculates the L2 MNE inverse solution for given data
        :param data: instance of Raw data
        :param method: 'MNE' | 'dSPM' | 'sLORETA'
            Use mininum norm, dSPM or sLORETA.
        :return: source estimate of the raw data with inverse operator applied
        """
        fwd = mne.convert_forward_solution(self.fwd, surf_ori=True)  # Orient the forward operator with surface
        #  coordinates
        inv = mne.minimum_norm.make_inverse_operator(data.info, fwd, self.cov, loose=self.loose, depth=self.depth)
        stc = apply_inverse_raw(data, inv, lambda2=self.lambda2, method=method)
        return stc

    # TODO: put this function outside of the object scope
    def data_func(self, times, n=0):
        """Generate time-staggered sinusoids at harmonics at 5-15Hz"""
        # According to Nyquist sampling theorem we need at least 2xf samples; let that be e.g. 2 x 15 = 30
        # TODO: Introduce distinct time courses for given activations!
        hz = 10
        n_samp = len(times)
        window = np.zeros(n_samp)
        start = 0
        stop = self.samples_per_dipole
        window[start:stop] = 1.
        n += 1
        data = 25e-9 * np.sin(2. * np.pi * hz * n * times)
        data *= window
        return data

    def __create_data_template(self):

        data_template = self.orig_data.copy()
        while len(data_template.times) < \
                (self.samples_per_dipole * (self.n_simulations + self.spacing_t_steps) *
                     (1 + self.empty_signal)):  # Expand raw_data template until it fits in time series
            data_template = mne.concatenate_raws([data_template, data_template])
        # Remove all projections from the data template
        data_template.info['bads'] = self.orig_data.info['bads']
        data_template.info["projs"] = []
        return data_template

    # TODO: implement these

    def simulate_evokeds(self, data):
        iir_filter = fit_iir_model_raw(data, order=5, picks=self.picks, tmin=60, tmax=180)[1]
        snr = SNR  # dB
        evoked = simulate_evoked(self.fwd, self.stc, self.info, self.cov, snr, iir_filter=iir_filter)
        return evoked

    def raw_preprocessing(self, raw, use_ica=True, use_ssp=True):

        # Filter

        raw.info['bads'] = self.bad_channels

        # Remove power-line noise
        raw.notch_filter(np.arange(60, 241, 60), picks=self.picks, filter_length='auto',
                         phase=self.phase)
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, phase=self.phase, fir_window=self.fir_window,
                   fir_design=self.fir_design)

        # Add EEG reference
        raw.set_eeg_reference()

        # TODO: some mechanism to control this
        if use_ica:
            # supported ica_channels = ['mag', 'grad', 'eeg', 'seeg', 'ecog', 'hbo', 'hbr', 'eog']
            ica_picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
            n_components = 25
            decim = 3

            # maximum number of components to reject
            n_max_ecg, n_max_eog = 3, 1  # here we don't expect horizontal EOG components

            # ica = run_ica(raw, n_components=0.95)

            ica = ICA(n_components=n_components, method='fastica', noise_cov=None)
            ica.fit(raw, decim=decim, picks=ica_picks, reject=self.reject)

            # generate ECG epochs use detection via phase statistics

            ecg_epochs = create_ecg_epochs(raw, reject=self.reject)

            ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')

            ecg_inds = ecg_inds[:n_max_ecg]
            ica.exclude += ecg_inds

            # detect EOG by correlation

            eog_inds, scores = ica.find_bads_eog(raw)

            eog_inds = eog_inds[:n_max_eog]
            ica.exclude += eog_inds

            ica.apply(raw)

        if use_ssp:
            if self.ecg:
                ecg_projs, _ = compute_proj_ecg(raw)
                raw.info['projs'] += ecg_projs
            if self.blink:
                eog_projs, _ = compute_proj_eog(raw)
                raw.info['projs'] += eog_projs

        raw.apply_proj()
        return raw

    def simulate_raw_data(self):
        """
        Simulates raw data
        """

        def expand_sim_stc_data(orig_times, t_index, data, interval):
            template = np.zeros((data.shape[0], len(orig_times)))
            template[:, t_index:t_index+interval] = data
            return template

        print("Generating simulated raw data...")

        if not self.data_template:
            self.data_template = self.__create_data_template()

        if self.n_dipoles == 1:
            time_steps = int(self.samples_per_dipole * self.spacing_t_steps * (1 + self.empty_signal))
        else:
            time_steps = int(self.samples_per_dipole * (
                self.n_simulations + self.spacing_t_steps) * (1 + self.empty_signal))

        data_template = self.data_template.copy()
        data_template = data_template.crop(0., data_template.times[time_steps])  # Crop a suitable piece of data
        t_delta = np.mean(np.gradient(data_template.times))
        times = np.arange(time_steps) * t_delta
        sim_stc = simulate_sparse_stc(
            self.src, n_dipoles=0, times=times, data_fun=self.data_func)  # Create empty time series

        s = 0
        lh_labels = mne.Label(self.lh_labels, hemi="lh")
        rh_labels = mne.Label(self.rh_labels, hemi="rh")

        lh_labels.values = np.zeros(lh_labels.values.shape)
        rh_labels.values = np.zeros(rh_labels.values.shape)

        # TODO: selection of hemisphere(s)
        self.times_per_dipole = int(self.samples_per_dipole * (1 + self.empty_signal))
        if self.n_dipoles == 1:
            print("Simulating 1 dipole in each vertex of the right hemisphere.")

            for rh in range(0, len(rh_labels.values)):
                template = sim_stc.copy()
                #lh_labels2 = lh_labels
                #lh_labels2.values[rh] = 1.
                rh_labels2 = rh_labels
                rh_labels2.values[rh] = 1.
                sim_stc2 = simulate_sparse_stc(self.src, n_dipoles=1, times=times[s:s+self.times_per_dipole],
                                               data_fun=self.data_func, labels=[rh_labels2])
                # TODO: use labels for deterministically simulating all vertices
                sim_stc.expand(sim_stc2.vertices), sim_stc2.expand(sim_stc.vertices), template.expand(sim_stc2.vertices)
                template.data[:, :] = expand_sim_stc_data(times, s, sim_stc2.data, self.times_per_dipole)
                sim_stc += template
                s += self.times_per_dipole

        else:
            print("Simulating up to %s dipoles on the right hemisphere." % self.n_dipoles)

            for t in range(0, len(times), self.times_per_dipole):
                template = sim_stc.copy()

                dipoles = random.randint(2, self.n_dipoles)

                label_indices = sorted(random.sample(range(0, len(rh_labels)), dipoles))
                chosen_label = rh_labels
                chosen_labels = []

                for l in label_indices:

                    chosen_label.values[l] = 1.
                    chosen_labels.append(chosen_label)
                    chosen_label.values[l] = 0.

                # TODO there may be a bug in MNE-Python 0.14.1 where forward model dipoles are mapped to non-unique
                # source space vertices, e.g. in this case rh vertices 298 and 411 both map to source space vertex
                # 112071 which raises an error in source_estimate.py at line 434

                try:
                    sim_stc2 = simulate_sparse_stc(self.src, n_dipoles=dipoles, times=times[t:t+self.times_per_dipole],
                                                   data_fun=self.data_func, labels=chosen_labels)
                except ValueError:
                    s -= self.times_per_dipole
                    continue
                sim_stc.expand(sim_stc2.vertices), sim_stc2.expand(sim_stc.vertices), template.expand(sim_stc2.vertices)
                template.data[:, :] = expand_sim_stc_data(times, t, sim_stc2.data, self.times_per_dipole)
                sim_stc.data[:, :] = sim_stc.data + template.data
                t += 1
                if s >= self.n_simulations:
                    break

        # Remove unnecessary zeros from series
        # sim_stc._data = sim_stc.data[:, ~np.all(abs(sim_stc.data) < 1e-20, axis=0)]  # TODO: add some zeroes to compensate?
        #sim_stc.times = sim_stc.times[:sim_stc.data.shape[1]]

        #template_times = np.arange(sim_stc.data.shape[1]) * (t_delta * (1 + self.empty_signal))
        #data_template = data_template.crop(0., template_times[-1])

        self.sim_data = simulate_raw(data_template, sim_stc, self.trans_path, self.src, self.bem_path,
                                     cov='simple', iir_filter=self.iir_filter, ecg=self.ecg, blink=self.blink,
                                     n_jobs=CPU_THREADS, verbose=self.verbose, use_cps=True)
        # self.cov = self.compute_covariance(self.simulate_evokeds(self.sim_data))
        self.sim_data = self.raw_preprocessing(self.sim_data)
        self.sim_stc = sim_stc
        self.cov = mne.compute_raw_covariance(self.sim_data, reject=self.reject, n_jobs=CPU_THREADS)
        return self.sim_data, self.sim_stc
