from __future__ import unicode_literals

from config import HEMISPHERE, NETWORK_TYPE, FINAL_WEIGHT_NAME, WEIGHT_PATH, N_DIPOLES, SNR
from parameters.parameters import get_simulation_model_kwargs
from os.path import join
from network.network_caller import NetworkCaller

# NOTE: This is currently only supported by the U-Net model.


def main():

    sim_kwargs = get_simulation_model_kwargs(N_DIPOLES, SNR)
    samples = sim_kwargs['samples_per_dipole']

    if N_DIPOLES == 1:
        n_tests = N_DIPOLES * samples


    print("Training neural network model for hemisphere: %s" % HEMISPHERE)

    if NETWORK_TYPE == 'unet':
        net = NetworkCaller(HEMISPHERE)
        weight_path = join(WEIGHT_PATH, 'model.cpkt')
        net.predict(weights=weight_path, n_tests=45)

    else:
        raise ValueError("Network type %s not understood" % NETWORK_TYPE)

if __name__ == "__main__":
    main()
