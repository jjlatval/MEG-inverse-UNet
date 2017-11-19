from __future__ import unicode_literals
from ConfigParser import SafeConfigParser
from config import SIMULATION_MODEL_CONFIG, DATA_PATH, SUBJECTS_DIR, SUBJECT_NAME, RAW_PATH, \
    BEM_PATH, COV_PATH, TRANS_PATH, SRC_PATH


def default_params():
    return {'simulation_model': {'data_path': DATA_PATH, 'subjects_dir': SUBJECTS_DIR,
                                 'subject_name': SUBJECT_NAME, 'raw_path': RAW_PATH,
                                 'bem_path': BEM_PATH, 'cov_path': COV_PATH,
                                 'trans_path': TRANS_PATH, 'src_path': SRC_PATH}}


def create_config(params=None):
    """
    Creates a config.ini file for the simulation_model
    :param params: Dict
        Dictionary of parameters
    :return:
    """
    if not params:
        params = default_params()
    config = SafeConfigParser()
    for key in params.keys():
        config.add_section(key)
        for item in params[key].items():
            config.set(key, item[0], item[1])

    with open(SIMULATION_MODEL_CONFIG, 'w') as f:
        config.write(f)
