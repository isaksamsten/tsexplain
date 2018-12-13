import numpy as np


def differences(a, b, axis=0, tolerance=1.e-8):
    """The number of time points that differ between `a` and `b`

    :param a: the first time series
    :param b: the second
    :param axis: axis to compare
    :returns: the fraction of differences

    """
    c = np.isclose(a, b, atol=tolerance)
    return np.mean(1 - np.sum(c, axis=axis) / a.shape[-1])


def failures(a):
    """The number of failed transformations

    :param a: the array of transformed time series
    :returns: the number of failures
    """
    return np.sum(np.all(np.isnan(a), axis=1, keepdims=True))


def cost(a, b):
    """Compute the cost of transformation

    :param a: first time series
    :param b: second time series
    :returns: the cost
    """
    return np.nanmean(np.linalg.norm(a - b, axis=1))
