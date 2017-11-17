from copy import deepcopy

import numpy as np
import scipy as sp

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.datautil.signalproc import bandpass_cnt
from braindecode.mne_ext.signalproc import mne_apply


def bandpass_mne(cnt, low_cut_hz, high_cut_hz, filt_order=3, axis=0):
    return mne_apply(lambda data: bandpass_cnt(
        data.T, low_cut_hz, high_cut_hz, fs=cnt.info['sfreq'],
        filt_order=filt_order, axis=axis).T,
              cnt)


def select_trials(dataset, inds):
    if hasattr(dataset.X, 'ndim'):
        # numpy array
        new_X = np.array(dataset.X)[inds]
    else:
        # list
        new_X = [dataset.X[i] for i in inds]
    new_y = np.asarray(dataset.y)[inds]
    return SignalAndTarget(new_X, new_y)


def select_classes_cnt(cnt, class_numbers):
    cnt = deepcopy(cnt)
    events = cnt.info['events']
    new_events = [ev for ev in events if (ev[2] - ev[1]) in class_numbers]
    cnt.info['events'] = np.array(new_events)
    return cnt


def select_classes(dataset, class_numbers):
    wanted_inds = [i_trial for i_trial, y in enumerate(dataset.y)
                   if y in class_numbers]
    return select_trials(dataset, wanted_inds)


def select_trials_cnt(cnt, inds):
    cnt = deepcopy(cnt)
    assert np.all([i in np.arange(len(cnt.info['events'])) for i in inds])
    events = cnt.info['events']
    new_events = [ev for i_trial, ev in enumerate(events) if i_trial in inds]
    cnt.info['events'] = np.array(new_events)
    return cnt

def concatenate_channels(datasets):
    all_X = [dataset.X for dataset in datasets]
    new_X = np.concatenate(all_X, axis=1)
    new_y = datasets[0].y
    for dataset in datasets:
        assert np.array_equal(dataset.y, new_y)
    return SignalAndTarget(new_X, new_y)


def extract_all_start_codes(name_to_start_codes):
    all_start_codes = []
    for val in name_to_start_codes.values():
        if hasattr(val, '__len__'):
            all_start_codes.extend(val)
        else:
            all_start_codes.append(val)
    return all_start_codes


def calculate_csp(epo, classes=None, average_trial_covariance=False):
    """Calculate the Common Spatial Pattern (CSP) for two classes.
    Now with pattern computation as in matlab bbci toolbox
    https://github.com/bbci/bbci_public/blob/c7201e4e42f873cced2e068c6cbb3780a8f8e9ec/processing/proc_csp.m#L112

    This method calculates the CSP and the corresponding filters. Use
    the columns of the patterns and filters.
    Examples
    --------
    Calculate the CSP for the first two classes::
    >>> w, a, d = calculate_csp(epo)
    >>> # Apply the first two and the last two columns of the sorted
    >>> # filter to the data
    >>> filtered = apply_spatial_filter(epo, w[:, [0, 1, -2, -1]])
    >>> # You'll probably want to get the log-variance along the time
    >>> # axis, this should result in four numbers (one for each
    >>> # channel)
    >>> filtered = np.log(np.var(filtered, 0))
    Select two classes manually::
    >>> w, a, d = calculate_csp(epo, [2, 5])
    Parameters
    ----------
    epo : epoched Data object
        this method relies on the ``epo`` to have three dimensions in
        the following order: class, time, channel
    classes : list of two ints, optional
        If ``None`` the first two different class indices found in
        ``epo.axes[0]`` are chosen automatically otherwise the class
        indices can be manually chosen by setting ``classes``
    Returns
    -------
    v : 2d array
        the sorted spatial filters
    a : 2d array
        the sorted spatial patterns. Column i of a represents the
        pattern of the filter in column i of v.
    d : 1d array
        the variances of the components
    Raises
    ------
    AssertionError :
        If:
          * ``classes`` is not ``None`` and has less than two elements
          * ``classes`` is not ``None`` and the first two elements are
            not found in the ``epo``
          * ``classes`` is ``None`` but there are less than two
            different classes in the ``epo``
    See Also
    --------
    :func:`apply_spatial_filter`, :func:`apply_csp`, :func:`calculate_spoc`
    References
    ----------
    http://en.wikipedia.org/wiki/Common_spatial_pattern
    """
    if classes is None:
        # automagically find the first two different classidx
        # we don't use uniq, since it sorts the classidx first
        # first check if we have a least two diffeent idxs:
        unique_classes = np.unique(epo.y)
        assert len(unique_classes) == 2
        cidx1 = unique_classes[0]
        cidx2 = unique_classes[1]
    else:
        assert (len(classes) == 2 and
                classes[0] in epo.y and
                classes[1] in epo.y)
        cidx1 = classes[0]
        cidx2 = classes[1]
    epo1 = select_classes(epo, [cidx1])
    epo2 = select_classes(epo, [cidx2])
    if average_trial_covariance:
        # computing c1 as mean covariance  of trial covariances:
        c1 = np.mean([np.cov(x) for x in epo1.X], axis=0)
        c2 = np.mean([np.cov(x) for x in epo2.X], axis=0)
    else:
        # we need a matrix of the form (channels, observations) so we stack trials
        # and time per channel together
        x1 = np.concatenate(epo1.X, axis=1)
        x2 = np.concatenate(epo2.X, axis=1)
        # compute covariance matrices of the two classes
        c1 = np.cov(x1)
        c2 = np.cov(x2)
    # solution of csp objective via generalized eigenvalue problem
    # in matlab the signature is v, d = eig(a, b)

    d, v = sp.linalg.eigh(c2, c1 + c2)
    d = d.real
    # make sure the eigenvalues and -vectors are correctly sorted
    indx = np.argsort(d)
    # reverse
    indx = indx[::-1]
    d = d.take(indx)
    v = v.take(indx, axis=1)

    # Now compute patterns
    # old pattern computation
    # a = sp.linalg.inv(v).transpose()
    c_avg = (c1 + c2) / 2.0

    # compare
    # https://github.com/bbci/bbci_public/blob/c7201e4e42f873cced2e068c6cbb3780a8f8e9ec/processing/proc_csp.m#L112
    # with W := v
    v_with_cov = np.dot(c_avg, v)
    source_cov = np.dot(np.dot(v.T, c_avg), v)
    # matlab-python comparison
    """
    v_with_cov = np.array([[1,2,-2],
             [3,-2,4],
             [5,1,0.3]])

    source_cov = np.array([[1,2,0.5],
                  [2,0.6,4],
                  [0.5,4,2]])

    sp.linalg.solve(source_cov.T, v_with_cov.T).T
    # for matlab
    v_with_cov = [[1,2,-2],
                 [3,-2,4],
                 [5,1,0.3]]

    source_cov = [[1,2,0.5],
                  [2,0.6,4],
                  [0.5,4,2]]
    v_with_cov / source_cov"""

    a = sp.linalg.solve(source_cov.T, v_with_cov.T).T
    return v, a, d


def apply_csp_fast(epo, filt, columns=[0, -1]):
    """Apply the CSP filter.

    Apply the spacial CSP filter to the epoched data.

    Parameters
    ----------
    epo : epoched ``Data`` object
        this method relies on the ``epo`` to have three dimensions in
        the following order: class, time, channel
    filt : 2d array
        the CSP filter (i.e. the ``v`` return value from
        :func:`calculate_csp`)
    columns : array of ints, optional
        the columns of the filter to use. The default is the first and
        the last one.

    Returns
    -------
    epo : epoched ``Data`` object
        The channels from the original have been replaced with the new
        virtual CSP channels.

    Examples
    --------

    >>> w, a, d = calculate_csp(epo)
    >>> epo = apply_csp_fast(epo, w)

    See Also
    --------
    :func:`calculate_csp`
    :func:`apply_csp`

    """
    f = filt[:, columns]
    filtered = []
    for trial_i in range(len(epo.X)):
        # time x filters
        this_filtered = np.dot(epo.X[trial_i].T, f)
        # to filters x time
        filtered.append(this_filtered.T)
    return SignalAndTarget(filtered, epo.y)


def apply_csp_var_log(epo, filters, columns):
    csp_filtered = apply_csp_fast(epo, filters, columns)
    # 1 is t
    csp_filtered.X = np.array([np.log(np.var(trial, axis=1))
                              for trial in csp_filtered.X])
    return csp_filtered