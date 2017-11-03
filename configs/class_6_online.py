import os

os.sys.path.insert(0, '/home/schirrmr/braindecode/code/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/fbcsp/')
import logging
import time
from collections import OrderedDict
from copy import copy

import numpy as np
from numpy.random import RandomState

from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact, save_npy_artifact
from braindecode.datasets.bbci import  BBCIDataset

from fbcsp.experiment import CSPExperiment

log = logging.getLogger(__name__)
log.setLevel('DEBUG')

def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': './data/models/fbcsp/6-class/avg-cov/',
    }]
    filename_params = [
        {
            'filename': '/data/schirrmr/schirrmr/offline-6-class-cabin/AnTiCUO1_1-4_250Hz.BBCI.mat',
        },
        {
            'filename': '/data/schirrmr/schirrmr/offline-6-class-cabin/FeHeCUO1_1-8_250Hz.BBCI.mat',
        },]
    filterbank_params = [
        {
            'min_freq': 1,
            'max_freq': 38,
            'low_width': 6,
            'high_width': 8,
            'high_overlap': 4,
            'last_low_freq': 10,
            'low_overlap': 3,
            'n_top_bottom_csp_filters': 5,
            'n_selected_features': 20
        },
        {
            'min_freq': 1,
            'max_freq': 118,
            'low_width': 6,
            'high_width': 8,
            'high_overlap': 4,
            'last_low_freq': 10,
            'low_overlap': 3,
            'n_top_bottom_csp_filters': 5,
            'n_selected_features': 20
        },
    ]
    sensor_params = [
        {
            'sensors': 'all',
        },
        {
            'sensors': 'C_sensors'
        }]


    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        filename_params,
        filterbank_params,
        sensor_params
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def run_exp(
        filename,
        min_freq,
        max_freq,
        low_width,
        high_width,
        high_overlap,
        last_low_freq,
        low_overlap,
        n_top_bottom_csp_filters,
        n_selected_features,
        sensors):
    if sensors == 'all':
        cnt = BBCIDataset(filename).load()
    else:
        assert sensors == 'C_sensors'
        sensor_names = [
            'FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5',
            'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
            'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h',
            'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h',
            'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h', 'CCP1h',
            'CCP2h', 'CPP1h', 'CPP2h']
        cnt = BBCIDataset(filename, load_sensor_names=sensor_names).load()


    cnt = cnt.drop_channels(['STI 014'])
    name_to_start_codes = OrderedDict([('Left Hand', [1]), ('Foot', [2],),
                                       ('Right Hand', [3]), ('Word', [4]),
                                       ('Mental Rotation', 5),
                                       ('Rest', 6)])
    name_to_stop_codes = OrderedDict([('Left Hand', [10]), ('Foot', [20],),
                                      ('Right Hand', [30]), ('Word', [40]),
                                      ('Mental Rotation', 50),
                                      ('Rest', 60)])
    csp_experiment = CSPExperiment(
        cnt, name_to_start_codes,
        epoch_ival_ms=[500, 0],
        name_to_stop_codes=name_to_stop_codes,
        min_freq=min_freq,
        max_freq=max_freq,
        last_low_freq=last_low_freq,
        low_width=low_width,
        high_width=high_width,
        low_overlap=low_overlap,
        high_overlap=high_overlap,
        filt_order=4,
        n_folds=5,
        n_top_bottom_csp_filters=n_top_bottom_csp_filters,
        # this number times two will be number of csp filters per filterband before feature selection
        n_selected_filterbands=None,  # how many filterbands to select?
        n_selected_features=n_selected_features,
        # how many Features to select with the feature selection?
        forward_steps=2,  # feature selection param
        backward_steps=1,  # feature selection param
        stop_when_no_improvement=False,  # feature selection param
        only_last_fold=True,
        # Split into number of folds, but only run the last fold (i.e. last fold as test fold)?
        restricted_n_trials=None,
        # restrict to certain number of _clean_ trials?
        shuffle=False,  # shuffle or do blockwise folds?
        low_bound=0.)
    csp_experiment.run()
    return csp_experiment


def run(
        ex,
        filename,
        min_freq,
        max_freq,
        low_width,
        high_width,
        high_overlap,
        last_low_freq,
        low_overlap,
        n_top_bottom_csp_filters,
        n_selected_features,
        sensors):
    import sys
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                     level=logging.DEBUG, stream=sys.stdout)
    start_time = time.time()
    ex.info['finished'] = False

    csp_experiment = run_exp(
        filename,
        min_freq,
        max_freq,
        low_width,
        high_width,
        high_overlap,
        last_low_freq,
        low_overlap,
        n_top_bottom_csp_filters,
        n_selected_features,
        sensors)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info['train_misclass'] = 1 - np.mean(
        csp_experiment.multi_class.train_accuracy)
    ex.info['test_misclass'] = 1 - np.mean(
        csp_experiment.multi_class.test_accuracy)
    ex.info['runtime'] = run_time
    ex.info['finished'] = True
    results = dict()
    results['binary_train_acc'] = csp_experiment.binary_csp.train_accuracy
    results['binary_test_acc'] = csp_experiment.binary_csp.test_accuracy
    results[
        'filterbank_train_acc'] = csp_experiment.filterbank_csp.train_accuracy
    results['filterbank_test_acc'] = csp_experiment.filterbank_csp.test_accuracy
    results['multi_train_acc'] = csp_experiment.multi_class.train_accuracy
    results['multi_test_acc'] = csp_experiment.multi_class.test_accuracy
    results[
        'train_pred_labels'] = csp_experiment.multi_class.train_predicted_labels
    results['train_labels'] = csp_experiment.multi_class.train_labels
    results[
        'test_pred_labels'] = csp_experiment.multi_class.test_predicted_labels
    results['test_labels'] = csp_experiment.multi_class.test_labels
    save_pkl_artifact(ex, results, 'results.pkl')
