from __future__ import division
import numpy as np

def zscore(proj, samples):
    """
    :param proj:
    :param samples:
    :return zscore of samples:
    """
    return (proj - samples.mean(axis=1)) / samples.std(axis=1)

def onetail(proj, samples):
    """
    :param proj:
    :param samples:
    :return pvalue: fraction of times the node projection is greater than random samples
    """
    return (proj[:,None] > samples).sum(axis=1) / samples.shape[0]

def istvan(proj, samples):
    """
    :param proj:
    :param samples:
    :return istvan:
    """
    raise NotImplementedError