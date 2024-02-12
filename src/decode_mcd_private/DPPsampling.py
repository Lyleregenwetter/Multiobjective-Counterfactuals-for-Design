# Taken from: https://github.com/DamianStraszak/FairDiverseDPPSampling

import numpy as np
import numpy.linalg as la
import numpy.random as nprand
import random
# from dppy.finite_dpps import FiniteDPP


# noinspection SpellCheckingInspection,PyTypeChecker
def uniform_sample(elems, k):
    n = len(elems)
    l = range(0, n)
    random.shuffle(l)
    return [elems[i] for i in l[:k]]

def pure_greedy(matrix, n):
    indices = np.unravel_index(np.argmax(matrix), matrix.shape) # picks two indices corresponding to max value in matrix
    indices = list(indices)
    while len(indices) < n:
        subset = matrix[indices, :]
        idx = np.argmax(np.min(subset, axis=0))
        indices.append(idx)
    return indices

# noinspection SpellCheckingInspection,PyTypeChecker
def kDPPGreedySample(Y, k):
    X = Y.copy()
    n = int(X.shape[0])
    S = []
    while len(S) < k:
        multinom = [0] * n
        for j in range(n):
            multinom[j] = pow(la.norm(X[j, :]), 2)
        multinomSum = sum(multinom)
        if multinomSum < 1e-9:
            raise ValueError('PartitionDPP sampler failed -- dimension of data too low.')
        multinom = multinom / multinomSum
        ind = nprand.multinomial(1, multinom)
        ind = np.where(ind == 1)
        ind = ind[0][0]
        S.append(ind)
        Xind = X[ind, :].copy()
        normInd = pow(la.norm(X[ind, :]), 2)
        for j in range(n):
            X[j, :] = X[j, :] - (np.dot(Xind, np.transpose(X[j, :])) / normInd) * Xind
    return S


# noinspection SpellCheckingInspection,PyTypeChecker
def PartitionDPPGreedySample(Y, kvec, Pvec):
    X = Y.copy()
    n = int(X.shape[0])
    S = []
    k = sum(kvec)
    p = len(kvec)
    cvec = [0] * p
    for i in range(k):
        multinom = [0] * n
        for j in range(n):
            if cvec[Pvec[j]] + 1 <= kvec[Pvec[j]]:
                multinom[j] = pow(la.norm(X[j, :]), 2) * ((kvec[Pvec[j]] - cvec[Pvec[j]]) * 1.0 / kvec[Pvec[j]])
            else:
                multinom[j] = 0
        multinomSum = sum(multinom)
        if multinomSum < 1e-9:
            raise ValueError('PartitionDPP sampler failed -- dimension of data too low.')
        multinom = multinom / multinomSum
        ind = np.argmin(multinom)
        # ind = nprand.multinomial(1, multinom)
        # ind = np.where(ind == 1)
        # ind = ind[0][0]
        S.append(ind)
        cvec[Pvec[ind]] += 1
        Xind = X[ind, :].copy()
        normInd = pow(la.norm(X[ind, :]), 2)
        for j in range(n):
            X[j, :] = X[j, :] - (np.dot(Xind, np.transpose(X[j, :])) / normInd) * Xind
    return S


# noinspection SpellCheckingInspection,PyTypeChecker
def kiDPPGreedySample(Y, kvec, Pvec):
    X = Y.copy()
    n = int(X.shape[0])
    p = len(kvec)
    P = [[] for _ in kvec]
    for i in range(n):
        P[Pvec[i]].append(i)
    S = []
    for i in range(p):
        M = P[i]
        X0 = X[M, :].copy()
        S0 = kDPPGreedySample(X0, kvec[i])
        S = S + [M[e] for e in S0]
    ksum = 0
    for ki in kvec:
        ksum += ki
    if len(S) != ksum:
        raise ValueError('PartitionDPP sampler failed -- dimension of data too low.')
    return S


    from numpy import linalg as la
import numpy as np


def kDPPExactSample(weighted_matrix, k):
    # weighted_matrix = nearestPD(weighted_matrix)
    print(weighted_matrix)
    print(min(np.linalg.eigvals(weighted_matrix)))
    DPP = FiniteDPP('likelihood', **{'L': weighted_matrix})
    DPP.sample_exact_k_dpp(size=k)
    samples_index = DPP.list_of_samples[0]
    return samples_index

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

