from __future__ import division
import numpy as np

def rw_kernel(A, nRw):
    return np.linalg.matrix_power(A / A.sum(axis=0), nRw)

def rwr_kernel(A, alpha):
    if alpha != 0.:
        Dinv = np.diag(1 / A.sum(axis=0))
        W = np.dot(A, Dinv)
        I = np.eye(A.shape[0])
        return alpha * np.linalg.inv((I - (1 - alpha) * W))
    else:
        return np.tile(A.sum(axis=1, keepdims=True) / A.sum(), [1, A.shape[0]])

def dsd_kernel(adjacency, nRw):
    from numpy.linalg import inv
    from scipy.spatial.distance import pdist, squareform
    n = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    p = adjacency / degree
    if nRw >= 0:
        c = np.eye(n)
        for i in xrange(nRw):
            c = np.dot(c, p) + np.eye(n)
        return squareform(pdist(c,metric='cityblock'))
    else:
        pi = degree / degree.sum()
        return squareform(pdist(inv(np.eye(n) - p - pi.T),metric='cityblock'))

def heat_kernel(A, t):
    D = np.diag(A.sum(axis=0))
    W = D - A
    return np.exp(-1 * t * W)

def istvan_kernel(A):
    raise NotImplementedError

