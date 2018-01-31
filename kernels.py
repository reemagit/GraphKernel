from __future__ import division
import numpy as np
from scipy import sparse

# Notice: The article references refer to the papers from which the kernel equations have been considered for
# implementation here, NOT the original paper where the kernel has been proposed.

# Random Walk (RW) kernel (Cowen et al., 2017)
def rw_kernel(A, nRw):
    Asp = sparse.csc_matrix(A / A.sum(axis=0))
    return np.asarray((Asp ** nRw).todense())

# Random Walk with Restart (RWR) kernel (Cowen et al., 2017)
def rwr_kernel(A, alpha):
    if alpha != 0.:
        Dinv = np.diag(1 / A.sum(axis=0))
        W = np.dot(A, Dinv)
        I = np.eye(A.shape[0])
        return alpha * np.linalg.inv((I - (1 - alpha) * W))
    else:
        return np.tile(A.sum(axis=1, keepdims=True) / A.sum(), [1, A.shape[0]])

# Diffusion State Distance (DSD) (Cowen et al., 2017)
# Note: this is a distance matrix, NOT a kernel (similarity)!
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

# Heat kernel (HK) (Cowen et al., 2017)
def heat_kernel(A, t):
    D = np.diag(A.sum(axis=0))
    W = D - A
    Wtexp = sparse.csc_matrix(-1 * t * W)
    return sparse.linalg.expm(Wtexp)

# Interconnectedness (ICN) kernel (Hsu et al., 2011)
def icn_kernel(A):
    Dinv = np.sqrt(A.sum(axis=0))
    Asp = sparse.csc_matrix(A)
    return np.asarray((Asp**2 + Asp * 2).todense()) / (Dinv[:,None] * Dinv)

def istvan_kernel(A):
    Dsqinv = sparse.csc_matrix(np.diag(1 / np.sqrt(A.sum(axis=0))))
    Asp = sparse.csc_matrix(A)
    return Asp * Dsqinv * Asp * Dsqinv
    #return np.dot(A, np.dot(Dsqinv, np.dot(A, Dsqinv)))