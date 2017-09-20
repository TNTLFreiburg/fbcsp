import numpy as np
from sklearn.covariance import LedoitWolf

def lda_train_scaled(fv, shrink=False):
    """Train the LDA classifier.

    Parameters
    ----------
    fv : ``Data`` object
        the feature vector must have 2 dimensional data, the first
        dimension being the class axis. The unique class labels must be
        0 and 1 otherwise a ``ValueError`` will be raised.
    shrink : Boolean, optional
        use shrinkage

    Returns
    -------
    w : 1d array
    b : float

    Raises
    ------
    ValueError : if the class labels are not exactly 0s and 1s

    Examples
    --------

    >>> clf = lda_train(fv_train)
    >>> out = lda_apply(fv_test, clf)

    See Also
    --------
    lda_apply

    """
    assert shrink is True
    assert fv.X.ndim == 2
    x = fv.X
    y = fv.y
    if len(np.unique(y)) != 2:
        raise ValueError('Should only have two unique class labels, instead got'
            ': {labels}'.format(labels=np.unique(y)))
    # Use sorted labels
    labels = np.sort(np.unique(y))
    mu1 = np.mean(x[y == labels[0]], axis=0)
    mu2 = np.mean(x[y == labels[1]], axis=0)
    # x' = x - m
    m = np.empty(x.shape)
    m[y == labels[0]] = mu1
    m[y == labels[1]] = mu2
    x2 = x - m
    # w = cov(x)^-1(mu2 - mu1)
    if shrink:
        estimator = LedoitWolf()
        covm = estimator.fit(x2).covariance_
    else:
        covm = np.cov(x2.T)
    w = np.dot(np.linalg.pinv(covm), (mu2 - mu1))

    #  From matlab bbci toolbox:
    # https://github.com/bbci/bbci_public/blob/fe6caeb549fdc864a5accf76ce71dd2a926ff12b/classification/train_RLDAshrink.m#L133-L134
    #C.w= C.w/(C.w'*diff(C_mean, 1, 2))*2;
    #C.b= -C.w' * mean(C_mean,2);
    w = (w / np.dot(w.T, (mu2 - mu1))) * 2
    b = np.dot(-w.T, np.mean((mu1, mu2), axis=0))
    assert not np.any(np.isnan(w))
    assert not np.isnan(b)
    return w, b


def lda_apply(fv, clf):
    """Apply feature vector to LDA classifier.

    Parameters
    ----------
    fv : ``Data`` object
        the feature vector must have a 2 dimensional data, the first
        dimension being the class axis.
    clf : (1d array, float)

    Returns
    -------

    out : 1d array
        The projection of the data on the hyperplane.

    Examples
    --------

    >>> clf = lda_train(fv_train)
    >>> out = lda_apply(fv_test, clf)


    See Also
    --------
    lda_train

    """
    x = fv.X
    w, b = clf
    return np.dot(x, w) + b
