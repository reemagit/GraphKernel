from __future__ import division
import numpy as np

def zscore(proj, samples=None, mean=None, std=None):
    """
    :param proj:
    :param samples:
    :param mean: mean of samples, ignored if samples argument is provided
    :param std: std of samples, ignored if samples argument is provided
    :return zscore of samples:
    """
    if not (samples is not None or (mean is not None and std is not None)):
        raise ValueError('Either samples argument or (mean,std) arguments pair has to be provided to zscore function')
    if samples is not None:
        return (proj - samples.mean(axis=1)) / samples.std(axis=1)
    else:
        return (proj - mean) / std

def onetail(proj, samples):
    """
    :param proj:
    :param samples:
    :return pvalue: fraction of times the node projection is greater than random samples
    """
    return (proj[:,None] > samples).sum(axis=1) / samples.shape[0]

def istvan(proj, samples=None, mean=None, std=None):
    """
    :param proj:
    :param samples:
    :param mean: mean of samples, ignored if samples argument is provided
    :param std: std of samples, ignored if samples argument is provided
    :return istvan:
    """
    raise NotImplementedError