import itertools
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.datautil.trial_segment import (
    create_signal_target_from_raw_mne)
from braindecode.mne_ext.signalproc import concatenate_raws_with_events
from .binary import BinaryCSP
from .filterbank import FilterbankCSP
import numpy as np
from numpy.random import RandomState
from .filterbank import generate_filterbank, filterbank_is_stable
import logging

from .multiclass import MultiClassWeightedVoting

log = logging.getLogger(__name__)

class CSPExperiment(object):
    """
        A Filter Bank Common Spatial Patterns Experiment.

        Parameters
        ----------
        cnt : Dataset
            The continuous recordings with events in info['events']
        name_to_start_codes: dict
            Dictionary mapping class names to marker numbers, e.g.
            {'1 - Correct': [31], '2 - Error': [32]}
        epoch_ival_ms : sequence of 2 floats
            The start and end of the trial in milliseconds with respect to the markers.
        min_freq : int
            The minimum frequency of the filterbank.
        max_freq : int
            The maximum frequency of the filterbank.
        last_low_freq : int
            The last frequency with the low width frequency of the filterbank.
        low_width : int
            The width of the filterbands in the lower frequencies.
        low_overlap : int
            The overlap of the filterbands in the lower frequencies.
        high_width : int
            The width of the filterbands in the higher frequencies.
        high_overlap : int
            The overlap of the filterbands in the higher frequencies.
        filt_order : int
            The filter order of the butterworth filter which computes the filterbands.
        n_folds : int
            How many folds. Also determines size of the test fold, e.g.
            5 folds imply the test fold has 20% of the original data.
        n_top_bottom_csp_filters : int
            Number of top and bottom CSP filters to select from all computed filters.
            Top and bottom refers to CSP filters sorted by their eigenvalues.
            So a value of 3 here will lead to 6(!) filters.
            None means all filters.
        n_selected_filterbands : int
            Number of filterbands to select for the filterbank.
            Will be selected by the highest training accuracies.
            None means all filterbands.
        n_selected_features : int
            Number of features to select for the filterbank.
            Will be selected by an internal cross validation across feature
            subsets.
            None means all features.
        forward_steps : int
            Number of forward steps to make in the feature selection,
            before the next backward step.
        backward_steps : int
            Number of backward steps to make in the feature selection,
            before the next forward step.
        stop_when_no_improvement: bool
            Whether to stop the feature selection if the interal cross validation
            accuracy could not be improved after an epoch finished
            (epoch=given number of forward and backward steps).
            False implies always run until wanted number of features.
        only_last_fold: bool
            Whether to train only on the last fold. 
            True implies a train-test split, where the n_folds parameter
            determines the size of the test fold.
            Test fold will always be at the end of the data (timewise).
        restricted_n_trials: int
            Take only a restricted number of the clean trials.
            None implies all clean trials.
        shuffle: bool
            Whether to shuffle the clean trials before splitting them into folds.
            False implies folds are time-blocks, True implies folds are random
            mixes of trials of the entire file.
    """
    def __init__(
            self,
            cnt,
            name_to_start_codes,
            epoch_ival_ms,
            name_to_stop_codes=None,
            min_freq=0,
            max_freq=48,
            last_low_freq=48,
            low_width=4,
            low_overlap=0,
            high_width=4,
            high_overlap=0,
            filt_order=3,
            n_folds=5,
            n_top_bottom_csp_filters=None,
            n_selected_filterbands=None,
            n_selected_features=None,
            forward_steps=2,
            backward_steps=1,
            stop_when_no_improvement=False,
            only_last_fold=False,
            restricted_n_trials=None,
            shuffle=False,
            low_bound=0.2,
            average_trial_covariance=True):
        local_vars = locals()
        del local_vars['self']
        self.__dict__.update(local_vars)
        self.filterbank_csp = None
        self.binary_csp = None
        self.filterbands = None

    def run(self):
        self.init_training_vars()
        log.info("Running Training...")
        self.run_training()

    def init_training_vars(self):
        self.filterbands = generate_filterbank(
            min_freq=self.min_freq,
            max_freq=self.max_freq, last_low_freq=self.last_low_freq, 
            low_width=self.low_width, low_overlap=self.low_overlap,
            high_width=self.high_width, high_overlap=self.high_overlap,
            low_bound=self.low_bound)
        assert filterbank_is_stable(
            self.filterbands, self.filt_order,
            self.cnt.info['sfreq']), (
                "Expect filter bank to be stable given filter order.")
        # check if number of selected features is not too large
        if self.n_selected_features is not None:
            n_spatial_filters = self.n_top_bottom_csp_filters
            if n_spatial_filters is None:
                n_spatial_filters = len(self.cnt.ch_names)
            n_max_features = len(self.filterbands) * n_spatial_filters
            assert n_max_features >= self.n_selected_features, (
                "Cannot select more features than will be originally created "
                "Originally: {:d}, requested: {:d}".format(
                    n_max_features, self.n_selected_features)
            )

        n_classes = len(self.name_to_start_codes)
        self.class_pairs = list(itertools.combinations(range(n_classes),2))
        # use only number of clean trials to split folds
        epo = create_signal_target_from_raw_mne(
            self.cnt, name_to_start_codes=self.name_to_start_codes,
            epoch_ival_ms=self.epoch_ival_ms,
            name_to_stop_codes=self.name_to_stop_codes)
        n_trials = len(epo.X)
        if self.restricted_n_trials is not None:
            if self.restricted_n_trials <= 1:
                n_trials = int(n_trials * self.restricted_n_trials)
            else:
                n_trials = min(n_trials, self.restricted_n_trials)
        rng = RandomState(903372376)
        folds = get_balanced_batches(n_trials, rng, self.shuffle,
                                     n_batches=self.n_folds)

        # remap to original indices in unclean set(!)
        # train is everything except fold
        # test is fold inds
        self.folds = [{'train': np.setdiff1d(np.arange(n_trials),fold),
             'test': fold}
                      for fold in folds]
        if self.only_last_fold:
            self.folds = self.folds[-1:]

    def run_training(self):
        self.binary_csp = BinaryCSP(
            self.cnt, self.filterbands,
            self.filt_order, self.folds, self.class_pairs, 
            self.epoch_ival_ms, self.n_top_bottom_csp_filters,
            marker_def=self.name_to_start_codes,
            name_to_stop_codes=self.name_to_stop_codes,
            average_trial_covariance=self.average_trial_covariance)
        self.binary_csp.run()
        log.info("Filterbank...")
        self.filterbank_csp = FilterbankCSP(self.binary_csp, 
            n_features=self.n_selected_features,
            n_filterbands=self.n_selected_filterbands,
            forward_steps=self.forward_steps,
            backward_steps=self.backward_steps,
            stop_when_no_improvement=self.stop_when_no_improvement)
        self.filterbank_csp.run()

        log.info("Multiclass...")
        self.multi_class = MultiClassWeightedVoting(
                                    self.binary_csp.train_labels_full_fold, 
                                    self.binary_csp.test_labels_full_fold,
                                    self.filterbank_csp.train_pred_full_fold,
                                    self.filterbank_csp.test_pred_full_fold,
                                    self.class_pairs)
        self.multi_class.run()


class CSPRetrain():
    """ CSP Retraining on existing filters computed previously."""
    def __init__(self, trainer_filename, n_selected_features="asbefore",
            n_selected_filterbands="asbefore",forward_steps=2,
            backward_steps=1, stop_when_no_improvement=False):
        self.trainer_filename = trainer_filename
        self.n_selected_features = n_selected_features
        self.n_selected_filterbands = n_selected_filterbands
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps
        self.stop_when_no_improvement = stop_when_no_improvement

    def run(self):
        log.info("Loading trainer...")
        self.trainer = np.load(self.trainer_filename)
        if self.n_selected_features == "asbefore":
            self.n_selected_features = self.trainer.filterbank_csp.n_features
        if self.n_selected_filterbands == "asbefore":
            self.n_selected_filterbands = self.trainer.filterbank_csp.n_filterbands
        # For later storage, remember selected features and filterbands
        # TODELAY: solve this more cleanly during saving or sth :)
        self.trainer.original_params['n_selected_features'] = \
            self.n_selected_features
        self.trainer.original_params['n_selected_filterbands'] = \
            self.n_selected_filterbands
        recreate_filterbank(self.trainer, self.n_selected_features,
            self.n_selected_filterbands, self.forward_steps,
            self.backward_steps, self.stop_when_no_improvement)
        
        log.info("Rerunning filterbank...")
        self.trainer.filterbank_csp.run()
        recreate_multi_class(self.trainer)
        log.info("Rerunning multiclass...")
        self.trainer.multi_class.run()


class TrainTestCSPExperiment(CSPExperiment):
    def __init__(
            self,train_cnt, test_cnt,
            name_to_start_codes,
            epoch_ival_ms,
            name_to_stop_codes=None,
            min_freq=0,
            max_freq=48,
            last_low_freq=48,
            low_width=4,
            low_overlap=0,
            high_width=4,
            high_overlap=0,
            filt_order=3,
            n_folds=None,
            n_top_bottom_csp_filters=None,
            n_selected_filterbands=None,
            n_selected_features=None,
            forward_steps=2,
            backward_steps=1,
            stop_when_no_improvement=False,
            only_last_fold=False,
            restricted_n_trials=None,
            shuffle=False,
            low_bound=0.2,
            average_trial_covariance=True):
        self.test_cnt = test_cnt
        super(TrainTestCSPExperiment, self).__init__(
            train_cnt,
            name_to_start_codes=name_to_start_codes,
            epoch_ival_ms=epoch_ival_ms,
            name_to_stop_codes=name_to_stop_codes,
            min_freq=min_freq,
            max_freq=max_freq,
            last_low_freq=last_low_freq,
            low_width=low_width,
            low_overlap=low_overlap,
            high_width=high_width,
            high_overlap=high_overlap,
            filt_order=filt_order,
            n_folds=n_folds,
            n_top_bottom_csp_filters=n_top_bottom_csp_filters,
            n_selected_filterbands=n_selected_filterbands,
            n_selected_features=n_selected_features,
            forward_steps=forward_steps,
            backward_steps=backward_steps,
            stop_when_no_improvement=stop_when_no_improvement,
            only_last_fold=only_last_fold,
            restricted_n_trials=restricted_n_trials,
            shuffle=shuffle,
            low_bound=low_bound,
            average_trial_covariance=average_trial_covariance)



    def run(self):
        # Actually not necessary to overwrite, just to make sure its stays
        # same for now, in case superclass changes run method
        self.init_training_vars()
        log.info("Running Training...")
        self.run_training()

    def init_training_vars(self):
        assert self.n_folds is None, "Cannot use folds on train test split"
        assert self.restricted_n_trials is None, "Not implemented yet"
        self.filterbands = generate_filterbank(min_freq=self.min_freq,
            max_freq=self.max_freq, last_low_freq=self.last_low_freq, 
            low_width=self.low_width, low_overlap=self.low_overlap,
            high_width=self.high_width, high_overlap=self.high_overlap,
            low_bound=self.low_bound)
        assert filterbank_is_stable(self.filterbands, self.filt_order, 
            self.cnt.info['sfreq']), (
                "Expect filter bank to be stable given filter order.")
        # check if number of selected features is not too large

        if self.n_selected_features is not None:
            n_spatial_filters = self.n_top_bottom_csp_filters
            if n_spatial_filters is None:
                n_spatial_filters = len(self.cnt.ch_names)
            n_max_features = len(self.filterbands) * n_spatial_filters
            assert n_max_features >= self.n_selected_features, (
                "Cannot select more features than will be originally created "
                "Originally: {:d}, requested: {:d}".format(
                    n_max_features, self.n_selected_features)
            )
        n_classes = len(self.name_to_start_codes)
        self.class_pairs = list(itertools.combinations(range(n_classes),2))

        train_epo = create_signal_target_from_raw_mne(
            self.cnt, name_to_start_codes=self.name_to_start_codes,
            epoch_ival_ms=self.epoch_ival_ms,
            name_to_stop_codes=self.name_to_stop_codes)
        n_train_trials = len(train_epo.X)
        test_epo = create_signal_target_from_raw_mne(
            self.test_cnt, name_to_start_codes=self.name_to_start_codes,
            epoch_ival_ms=self.epoch_ival_ms,
            name_to_stop_codes=self.name_to_stop_codes)
        n_test_trials = len(test_epo.X)

        train_fold = np.arange(n_train_trials)
        test_fold = np.arange(n_train_trials, n_train_trials+n_test_trials)
        self.folds = [{'train': train_fold, 'test': test_fold}]
        assert np.intersect1d(self.folds[0]['test'], 
            self.folds[0]['train']).size == 0
        # merge cnts!!
        self.cnt = concatenate_raws_with_events([self.cnt, self.test_cnt])


def recreate_filterbank(train_csp_obj, n_features, n_filterbands,
        forward_steps, backward_steps, stop_when_no_improvement):
    train_csp_obj.filterbank_csp = FilterbankCSP(train_csp_obj.binary_csp,
        n_features, n_filterbands, 
            forward_steps=forward_steps,
            backward_steps=backward_steps,
            stop_when_no_improvement=stop_when_no_improvement)


def recreate_multi_class(train_csp_obj):
    """ Assumes filterbank + possibly binary csp was rerun and
    recreates multi class weighted voting object 
    with new labels + predictions. """
    train_csp_obj.multi_class = MultiClassWeightedVoting(
        train_csp_obj.binary_csp.train_labels_full_fold, 
        train_csp_obj.binary_csp.test_labels_full_fold,
        train_csp_obj.filterbank_csp.train_pred_full_fold,
        train_csp_obj.filterbank_csp.test_pred_full_fold,
        train_csp_obj.class_pairs)
