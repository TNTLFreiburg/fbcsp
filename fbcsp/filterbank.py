from copy import deepcopy
import logging
import numpy as np
import scipy.signal

from braindecode.datautil.iterators import get_balanced_batches
from .signalproc import concatenate_channels, select_trials
from .lda import lda_train_scaled, lda_apply


log = logging.getLogger(__name__)


def generate_filterbank(min_freq, max_freq, last_low_freq,
                        low_width, low_overlap, high_width, high_overlap,
                        low_bound):
    # int checks probably not necessary?
    # since we are using np.arange now below, not range
    # assert isinstance(min_freq, int) or min_freq.is_integer()
    # assert isinstance(max_freq, int) or max_freq.is_integer()
    # assert isinstance(last_low_freq, int) or last_low_freq.is_integer()
    # assert isinstance(low_width, int) or low_width.is_integer()
    # assert isinstance(high_width, int) or high_width.is_integer()
    assert low_overlap < low_width, "overlap needs to be smaller than width"
    assert high_overlap < high_width, "overlap needs to be smaller than width"
    low_step = low_width - low_overlap
    assert (last_low_freq - min_freq) % low_step == 0, ("last low freq "
                                                        "needs to be exactly the center of a low_width filter band. "
                                                        " Close center: {:f}".format(
        last_low_freq - ((last_low_freq - min_freq) % low_step)))
    assert max_freq >= last_low_freq
    high_step = high_width - high_overlap
    # end of last low frequency  - low overlap should be lower bound
    # of high frequency filterbands
    # => this lower bound + high width/2 is first center of high filterbands
    high_start = last_low_freq + (low_width / 2.0)  - low_overlap + (
         high_width / 2.0)
    assert (max_freq == last_low_freq or
            (max_freq - high_start) % high_step == 0), ("max freq needs to be "
                                                        "exactly the center of a filter band "
                                                        " Close center: {:f}".format(
        max_freq - ((max_freq - high_start) % high_step)))
    # + low_step/2.0 to also include the last low freq
    # analogous for high_step/2.0
    low_centers = np.arange(min_freq, last_low_freq + low_step / 2.0, low_step)
    high_centers = np.arange(high_start, max_freq + high_step / 2.0, high_step)

    low_band = np.array([np.array(low_centers) - low_width / 2.0,
                         np.array(low_centers) + low_width / 2.0]).T
    # low_band = np.maximum(0.2, low_band)
    # try since tonio wanted it with 0
    low_band = np.maximum(low_bound, low_band)
    high_band = np.array([np.array(high_centers) - high_width / 2.0,
                          np.array(high_centers) + high_width / 2.0]).T
    filterbank = np.concatenate((low_band, high_band))
    return filterbank


def filter_is_stable(a):
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a)))
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a)) < 1)


def filterbank_is_stable(filterbank, filt_order, sampling_rate):
    nyq_freq = 0.5 * sampling_rate
    for low_cut_hz, high_cut_hz in filterbank:
        low = low_cut_hz / nyq_freq
        high = high_cut_hz / nyq_freq
        if low == 0:
            b, a = scipy.signal.butter(filt_order, high, btype='lowpass')
        else:  # low!=0
            b, a = scipy.signal.butter(filt_order, [low, high],
                                       btype='bandpass')
        if not filter_is_stable(a):
            return False
    return True


def get_freq_inds(filterbands, low_freq, high_freq):
    i_low = np.where(filterbands[:, 0] == low_freq)[0][0]
    i_high = np.where(filterbands[:, 1] == high_freq)[0][0]
    return i_low, i_high


class FilterbankCSP(object):
    def __init__(self, binary_csp, n_features=None, n_filterbands=None,
                 forward_steps=2, backward_steps=1, stop_when_no_improvement=False):
        self.binary_csp = binary_csp
        self.n_features = n_features
        self.n_filterbands = n_filterbands
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps
        self.stop_when_no_improvement = stop_when_no_improvement

    def run(self):
        self.select_filterbands()
        if self.n_features is not None:
            log.info("Run feature selection...")
            self.collect_best_features()
            log.info("Done.")
            # self.select_features()
        else:
            self.collect_features()
        self.train_classifiers()
        self.predict_outputs()

    def select_filterbands(self):
        n_all_filterbands = len(self.binary_csp.filterbands)
        if self.n_filterbands is None:
            self.selected_filter_inds = list(range(n_all_filterbands))
        else:
            # Select the filterbands with the highest mean accuracy on the
            # training sets
            mean_accs = np.mean(self.binary_csp.train_accuracy, axis=(1, 2))
            best_filters = np.argsort(mean_accs)[::-1][:self.n_filterbands]
            self.selected_filter_inds = best_filters

    def collect_features(self):
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        result_shape = (n_folds, n_class_pairs)
        self.train_feature = np.empty(result_shape, dtype=object)
        self.train_feature_full_fold = np.empty(result_shape, dtype=object)
        self.test_feature = np.empty(result_shape, dtype=object)
        self.test_feature_full_fold = np.empty(result_shape, dtype=object)

        bcsp = self.binary_csp  # just to make code shorter
        filter_inds = self.selected_filter_inds
        for fold_i in range(n_folds):
            for class_i in range(n_class_pairs):
                self.train_feature[fold_i, class_i] = concatenate_channels(
                    bcsp.train_feature[filter_inds, fold_i, class_i])
                self.train_feature_full_fold[fold_i, class_i] = (
                    concatenate_channels(
                    bcsp.train_feature_full_fold[filter_inds, fold_i, class_i]))
                self.test_feature[fold_i, class_i] = concatenate_channels(
                    bcsp.test_feature[filter_inds, fold_i, class_i]
                )
                self.test_feature_full_fold[fold_i, class_i] = (
                    concatenate_channels(
                    bcsp.test_feature_full_fold[filter_inds, fold_i, class_i]
                ))

    def collect_best_features(self):
        """ Selects features filterwise per filterband, starting with no features,
        then selecting the best filterpair from the bestfilterband (measured on internal
        train/test split)"""
        bincsp = self.binary_csp  # just to make code shorter
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        result_shape = (n_folds, n_class_pairs)
        self.train_feature = np.empty(result_shape, dtype=object)
        self.train_feature_full_fold = np.empty(result_shape, dtype=object)
        self.test_feature = np.empty(result_shape, dtype=object)
        self.test_feature_full_fold = np.empty(result_shape, dtype=object)
        self.selected_filters_per_filterband = np.empty(result_shape, dtype=object)
        for fold_i in range(n_folds):
            for class_pair_i in range(n_class_pairs):
                bin_csp_train_features = deepcopy(
                    bincsp.train_feature[
                        self.selected_filter_inds, fold_i, class_pair_i])
                bin_csp_train_features_full_fold = deepcopy(
                    bincsp.train_feature_full_fold[
                        self.selected_filter_inds,
                        fold_i, class_pair_i])
                bin_csp_test_features = deepcopy(bincsp.test_feature[
                    self.selected_filter_inds, fold_i, class_pair_i])
                bin_csp_test_features_full_fold = deepcopy(
                    bincsp.test_feature_full_fold[
                        self.selected_filter_inds, fold_i, class_pair_i])
                selected_filters_per_filt = self.select_best_filters_best_filterbands(
                    bin_csp_train_features, max_features=self.n_features,
                    forward_steps=self.forward_steps,
                    backward_steps=self.backward_steps,
                    stop_when_no_improvement=self.stop_when_no_improvement)
                self.train_feature[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_train_features, selected_filters_per_filt)
                self.train_feature_full_fold[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_train_features_full_fold, selected_filters_per_filt)

                self.test_feature[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_test_features, selected_filters_per_filt)
                self.test_feature_full_fold[fold_i, class_pair_i] = \
                    self.collect_features_for_filter_selection(
                        bin_csp_test_features_full_fold, selected_filters_per_filt)

                self.selected_filters_per_filterband[fold_i, class_pair_i] = \
                    selected_filters_per_filt

    @staticmethod
    def select_best_filters_best_filterbands(features, max_features,
                                             forward_steps, backward_steps, stop_when_no_improvement):
        assert max_features is not None, (
            "For now not dealing with the case that max features is unlimited")
        assert features[0].X.shape[1] % 2 == 0
        n_filterbands = len(features)
        n_filters_per_fb = features[0].X.shape[1] / 2
        selected_filters_per_band = [0] * n_filterbands
        best_selected_filters_per_filterband = None
        last_best_accuracy = -1
        # Run until no improvement or max features reached
        selection_finished = False
        while (not selection_finished):
            for _ in range(forward_steps):
                best_accuracy = -1  # lets try always taking a feature in each iteration
                for filt_i in range(n_filterbands):
                    this_filt_per_fb = deepcopy(selected_filters_per_band)
                    if (this_filt_per_fb[filt_i] == n_filters_per_fb):
                        continue
                    this_filt_per_fb[filt_i] = this_filt_per_fb[filt_i] + 1
                    all_features = FilterbankCSP.collect_features_for_filter_selection(
                        features, this_filt_per_fb)
                    # make 5 times cross validation...
                    test_accuracy = FilterbankCSP.cross_validate_lda(all_features)
                    if (test_accuracy > best_accuracy):
                        best_accuracy = test_accuracy
                        best_selected_filters_per_filterband = this_filt_per_fb
                selected_filters_per_band = best_selected_filters_per_filterband
            for _ in range(backward_steps):
                best_accuracy = -1  # lets try always taking a feature in each iteration
                for filt_i in range(n_filterbands):
                    this_filt_per_fb = deepcopy(selected_filters_per_band)
                    if (this_filt_per_fb[filt_i] == 0):
                        continue
                    this_filt_per_fb[filt_i] = this_filt_per_fb[filt_i] - 1
                    all_features = FilterbankCSP.collect_features_for_filter_selection(
                        features, this_filt_per_fb)
                    # make 5 times cross validation...
                    test_accuracy = FilterbankCSP.cross_validate_lda(all_features)
                    if (test_accuracy > best_accuracy):
                        best_accuracy = test_accuracy
                        best_selected_filters_per_filterband = this_filt_per_fb
                selected_filters_per_band = best_selected_filters_per_filterband

            selection_finished = 2 * np.sum(selected_filters_per_band) >= max_features
            if stop_when_no_improvement:
                # there was no improvement if accuracy did not increase...
                selection_finished = (selection_finished or
                                      best_accuracy <= last_best_accuracy)
            last_best_accuracy = best_accuracy
        return selected_filters_per_band

    @staticmethod
    def collect_features_for_filter_selection(features, filters_for_filterband):
        n_filters_per_fb = features[0].X.shape[1] // 2
        n_filterbands = len(features)
        # start with filters of first filterband...
        # then add others all together
        first_features = deepcopy(features[0])
        first_n_filters = filters_for_filterband[0]
        if first_n_filters == 0:
            first_features.X = first_features.X[:,0:0]
        else:
            first_features.X = first_features.X[:, list(range(first_n_filters)) +
               list(range(-first_n_filters, 0))]

        all_features = first_features
        for i in range(1, n_filterbands):
            this_n_filters = min(n_filters_per_fb, filters_for_filterband[i])
            if this_n_filters > 0:
                next_features = deepcopy(features[i])
                if this_n_filters == 0:
                    next_features.X = next_features.X[0:0]
                else:
                    next_features.X = next_features.X[
                     :, list(range(this_n_filters)) +
                        list(range(-this_n_filters, 0))]
                all_features = concatenate_channels(
                    (all_features, next_features))
        return all_features

    @staticmethod
    def cross_validate_lda(features):
        n_trials = features.X.shape[0]
        folds = get_balanced_batches(n_trials, rng=None, shuffle=False,
                                     n_batches=5)
        # make to train-test splits, fold is test part..
        folds = [(np.setdiff1d(np.arange(n_trials), fold),
                  fold) for fold in folds]
        test_accuracies = []
        for train_inds, test_inds in folds:
            train_features = select_trials(features, train_inds)
            test_features = select_trials(features, test_inds)
            clf = lda_train_scaled(train_features, shrink=True)
            test_out = lda_apply(test_features, clf)

            higher_class = np.max(test_features.y)
            true_0_1_labels_test = test_features.y == higher_class

            predicted_test = test_out >= 0
            test_accuracy = np.mean(true_0_1_labels_test == predicted_test)
            test_accuracies.append(test_accuracy)
        return np.mean(test_accuracies)

    def select_features(self):
        n_folds = len(self.train_feature)
        n_pairs = len(self.train_feature[0])
        n_features = self.n_features
        self.selected_features = np.ones((n_folds, n_pairs, n_features),
                                         dtype=np.int) * -1

        # Determine best features
        for fold_nr in range(n_folds):
            for pair_nr in range(n_pairs):
                features = self.train_feature[fold_nr][pair_nr]
                this_feature_inds = select_features(features.axes[0],
                                                    features.data, n_features=n_features)
                self.selected_features[fold_nr][pair_nr] = this_feature_inds
        assert np.all(self.selected_features >= 0) and np.all(self.selected_features <
                                                              self.train_feature[0][0].data.shape[1])
        # Only retain selected best features
        for fold_nr in range(n_folds):
            for pair_nr in range(n_pairs):
                this_feature_inds = self.selected_features[fold_nr][pair_nr]
                for feature_type in ['train_feature', 'train_feature_full_fold',
                                     'test_feature', 'test_feature_full_fold']:
                    features = self.__dict__[feature_type][fold_nr][pair_nr]
                    features.data = features.data[:, this_feature_inds]

    def train_classifiers(self):
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        self.clf = np.empty((n_folds, n_class_pairs),
                            dtype=object)
        for fold_i in range(n_folds):
            for class_i in range(n_class_pairs):
                train_feature = self.train_feature[fold_i, class_i]
                clf = lda_train_scaled(train_feature, shrink=True)
                self.clf[fold_i, class_i] = clf

    def predict_outputs(self):
        n_folds = len(self.binary_csp.folds)
        n_class_pairs = len(self.binary_csp.class_pairs)
        result_shape = (n_folds, n_class_pairs)
        self.train_accuracy = np.empty(result_shape, dtype=float)
        self.test_accuracy = np.empty(result_shape, dtype=float)
        self.train_pred_full_fold = np.empty(result_shape, dtype=object)
        self.test_pred_full_fold = np.empty(result_shape, dtype=object)
        for fold_i in range(n_folds):
            log.info("Fold Nr: {:d}".format(fold_i + 1))
            for class_i, class_pair in enumerate(self.binary_csp.class_pairs):
                clf = self.clf[fold_i, class_i]
                class_pair_plus_one = (np.array(class_pair) + 1).tolist()
                log.info("Class {:d} vs {:d}".format(*class_pair_plus_one))
                train_feature = self.train_feature[fold_i, class_i]
                train_out = lda_apply(train_feature, clf)
                true_0_1_labels_train = train_feature.y == class_pair[1]
                predicted_train = train_out >= 0
                # remove xarray wrapper with float( ...
                train_accuracy = float(np.mean(true_0_1_labels_train ==
                                         predicted_train))
                self.train_accuracy[fold_i, class_i] = train_accuracy

                test_feature = self.test_feature[fold_i, class_i]
                test_out = lda_apply(test_feature, clf)
                true_0_1_labels_test = test_feature.y == class_pair[1]
                predicted_test = test_out >= 0
                test_accuracy = float(np.mean(true_0_1_labels_test ==
                                              predicted_test))

                self.test_accuracy[fold_i, class_i] = test_accuracy

                train_feature_full_fold = self.train_feature_full_fold[fold_i, \
                                                                       class_i]
                train_out_full_fold = lda_apply(train_feature_full_fold, clf)
                self.train_pred_full_fold[fold_i, class_i] = train_out_full_fold
                test_feature_full_fold = self.test_feature_full_fold[fold_i, \
                                                                     class_i]
                test_out_full_fold = lda_apply(test_feature_full_fold, clf)
                self.test_pred_full_fold[fold_i, class_i] = test_out_full_fold

                log.info("Train: {:4.2f}%".format(train_accuracy * 100))
                log.info("Test:  {:4.2f}%".format(test_accuracy * 100))
