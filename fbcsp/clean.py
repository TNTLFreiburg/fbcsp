from collections import namedtuple
import logging
import itertools

import numpy as np

from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from .signalproc import select_classes_cnt, select_trials_cnt, \
    extract_all_start_codes

log = logging.getLogger(__name__)

CleanResult = namedtuple('CleanResult', ['rejected_i_chans',
    'rejected_i_trials',])


def restrict_cnt(cnt, classes, clean_trials, rejected_chan_names):
    cleaned_cnt = select_classes_cnt(cnt, classes,)
    cleaned_cnt = select_trials_cnt(cleaned_cnt, clean_trials,)
    cleaned_cnt = cleaned_cnt.drop_channels(rejected_chan_names)
    return cleaned_cnt


def clean_cnt(cnt, epoch_ival_ms, name_to_start_codes, cleaner):
    log.info("Cleaning...")
    epo = create_signal_target_from_raw_mne(
        cnt,
        name_to_start_codes=name_to_start_codes,
        epoch_ival_ms=epoch_ival_ms)
    clean_result = cleaner.clean(epo.X)
    markers = extract_all_start_codes(name_to_start_codes)
    clean_trials = np.setdiff1d(np.arange(len(epo.X)),
                                clean_result.rejected_i_trials)
    rejected_chan_names = [cnt.ch_names[i_chan]
                           for i_chan in clean_result.rejected_i_chans]


    log.info("Rejected channels: {:s}".format(
        str(rejected_chan_names)))
    log.info("#Clean trials:     {:d}".format(len(clean_trials)))
    log.info("#Rejected trials:  {:d}".format(
        len(clean_result.rejected_i_trials)))
    log.info("Fraction Clean:    {:.1f}%".format(
        100 * len(clean_trials) /
        (len(clean_trials) + len(clean_result.rejected_i_trials))))

    cleaned_cnt = restrict_cnt(
        cnt,
        markers,
        clean_trials,
        rejected_chan_names,)
    return cleaned_cnt


def apply_multiple_cleaners(cnt, epoch_ival_ms, name_to_start_codes, cleaners):
    cleaned_cnt = cnt
    for cleaner in cleaners:
        log.info("Apply {:s}...".format(cleaner.__class__.__name__))
        cleaned_cnt = clean_cnt(cleaned_cnt, epoch_ival_ms,
                                name_to_start_codes, cleaner)
    return cleaned_cnt


class MaxAbsTrialCleaner(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def clean(self, X, ignore_chans=False):
        # max abs over samples and channels
        trial_max = np.array([np.max(np.abs(x)) for x in X])
        rejected_trials = np.flatnonzero(trial_max > self.threshold)
        clean_result = CleanResult(rejected_i_chans=[],
                                   rejected_i_trials=rejected_trials,)
        return clean_result


class MaxAbsChannelCleaner(object):
    def __init__(self, threshold, fraction):
        self.threshold = threshold
        self.fraction = fraction

    def clean(self, X, ignore_chans=False):
        # max abs over samples and channels
        trial_max_per_chan = np.array([np.max(np.abs(x), axis=1) for x in X])
        above_threshold = trial_max_per_chan > self.threshold
        fraction_per_chan = np.mean(above_threshold, axis=0)
        rejected_i_chans = np.flatnonzero(fraction_per_chan > self.fraction)
        clean_result = CleanResult(rejected_i_chans=rejected_i_chans,
                                   rejected_i_trials=[],)
        return clean_result


class EOGMaxMinCleaner(object):
    def __init__(self, eog_cnt, epoch_ival_ms, name_to_start_codes,
                 threshold):
        self.eog_cnt = eog_cnt
        self.epoch_ival_ms = epoch_ival_ms
        self.name_to_start_codes = name_to_start_codes
        self.threshold = threshold

    def clean(self, X, ignore_chans=False):
        eog_epo = create_signal_target_from_raw_mne(
            self.eog_cnt,
            name_to_start_codes=self.name_to_start_codes,
            epoch_ival_ms=self.epoch_ival_ms)

        clean_result = MaxMinCleaner(self.threshold).clean(eog_epo.X)
        return clean_result


class MaxMinCleaner(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def clean(self, X):
        rejected_trials = compute_rejected_trials_max_min(X, self.threshold)
        clean_result = CleanResult(rejected_i_chans=[],
                                   rejected_i_trials=rejected_trials,)
        return clean_result


class VarCleaner(object):
    def __init__(self, whisker_percent, whisker_length):
        self.whisker_percent = whisker_percent
        self.whisker_length = whisker_length

    def clean(self, X, ignore_chans=False):
        variances = np.var(X, axis=2)

        rejected_i_chans, rejected_i_trials = (
            compute_rejected_channels_trials_by_variance(
                variances, self.whisker_percent, self.whisker_length,
                ignore_chans=ignore_chans))
        clean_result = CleanResult(rejected_i_chans=rejected_i_chans,
                                   rejected_i_trials=rejected_i_trials,)
        return clean_result


def compute_rejected_trials_max_min(X, threshold):
    # max min over time inside trials
    max_vals = np.max(X, axis=2)
    min_vals = np.min(X, axis=2)
    maxmin_diffs = max_vals - min_vals
    assert maxmin_diffs.ndim == 2  # trials x channels
    # from these diffs, take maximum over chans, since we throw out trials if any chan
    # is exceeding the limit
    maxmin_diffs = np.max(maxmin_diffs, axis=1)
    assert maxmin_diffs.ndim == 1  # just trials
    rejected_trials_max_min = np.flatnonzero(maxmin_diffs > threshold)
    return rejected_trials_max_min


def get_variance_threshold(variances, whisker_percent, whisker_length):
    """Get the threshold variance, above which variance is defined as an outlier/to be rejected."""
    low_percentiles, high_percentiles = np.percentile(variances, (
    whisker_percent, 100 - whisker_percent))
    threshold = high_percentiles + (
                                   high_percentiles - low_percentiles) * whisker_length

    return threshold


# test create set three trials, one trial has excessive variance, should be removed
# create set with three channels, one excessive variance, should be removed
# create set
def compute_rejected_channels_trials_by_variance(variances, whisker_percent,
                                                 whisker_length, ignore_chans):
    orig_chan_inds = range(variances.shape[1])
    orig_trials = range(variances.shape[0])
    good_chan_inds = np.copy(orig_chan_inds)
    good_trials = np.copy(orig_trials)

    # remove trials with excessive variances
    bad_trials = compute_excessive_outlier_trials(variances, whisker_percent,
                                                  whisker_length)
    good_trials = np.delete(good_trials, bad_trials, axis=0)
    variances = np.delete(variances, bad_trials, axis=0)

    # now remove channels (first)
    if not ignore_chans:
        no_further_rejections = False
        while not no_further_rejections:
            bad_chans = compute_outlier_chans(variances, whisker_percent,
                                              whisker_length)
            variances = np.delete(variances, bad_chans, axis=1)
            good_chan_inds = np.delete(good_chan_inds, bad_chans, axis=0)
            no_further_rejections = len(bad_chans) == 0

    # now remove trials (second)
    no_further_rejections = False
    while not no_further_rejections:
        bad_trials = compute_outlier_trials(variances, whisker_percent,
                                            whisker_length)
        good_trials = np.delete(good_trials, bad_trials, axis=0)
        variances = np.delete(variances, bad_trials, axis=0)
        no_further_rejections = len(bad_trials) == 0

    # remove unstable chans
    if not ignore_chans:
        bad_chans = compute_unstable_chans(variances, whisker_percent,
                                           whisker_length)
        good_chan_inds = np.delete(good_chan_inds, bad_chans, axis=0)

    rejected_chan_inds = np.setdiff1d(orig_chan_inds, good_chan_inds)
    rejected_trials = np.setdiff1d(orig_trials, good_trials)
    return rejected_chan_inds, rejected_trials


def compute_outlier_chans(variances, whisker_percent, whisker_length):
    num_trials = variances.shape[0]
    threshold = get_variance_threshold(variances, whisker_percent,
                                       whisker_length)
    above_threshold = variances > threshold
    # only remove any channels if more than 5 percent of trials across channels are exceeding the variance
    if (np.sum(above_threshold) > 0.05 * num_trials):
        fraction_of_all_outliers_per_chan = np.sum(above_threshold,
                                                   axis=0) / float(
            np.sum(above_threshold))
        chan_has_many_bad_trials = np.mean(above_threshold, axis=0) > 0.05
        chan_has_large_fraction_of_outliers = fraction_of_all_outliers_per_chan > 0.1
        bad_chans = np.logical_and(chan_has_large_fraction_of_outliers,
                                   chan_has_many_bad_trials)
        assert bad_chans.ndim == 1
        bad_chans = np.flatnonzero(bad_chans)
    else:
        bad_chans = []
    return bad_chans


def compute_unstable_chans(variances, whisker_percent, whisker_length):
    variance_of_variance = np.var(variances, axis=0)
    threshold = get_variance_threshold(variance_of_variance, whisker_percent,
                                       whisker_length)
    bad_chans = variance_of_variance > threshold
    bad_chans = np.flatnonzero(bad_chans)
    return bad_chans


def compute_outlier_trials(variances, whisker_percent, whisker_length):
    threshold = get_variance_threshold(variances, whisker_percent,
                                       whisker_length)
    above_threshold = variances > threshold
    trials_one_chan_above_threshold = np.any(above_threshold, axis=1)
    outlier_trials = np.flatnonzero(trials_one_chan_above_threshold)
    return outlier_trials


def compute_excessive_outlier_trials(variances, whisker_percent,
                                     whisker_length):
    # clean trials with "excessive variance":
    # trials, where 20 percent of chans are above
    # whisker determined threshold
    threshold = get_variance_threshold(variances, whisker_percent,
                                       whisker_length)
    above_threshold = variances > threshold
    fraction_chans_above_threshold = np.mean(above_threshold, axis=1)
    assert fraction_chans_above_threshold.ndim == 1
    outlier_trials = np.flatnonzero(fraction_chans_above_threshold > 0.2)
    return outlier_trials
