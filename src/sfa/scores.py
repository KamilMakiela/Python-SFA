# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:27:41 2026

@author: makielak
"""

import numpy as np
from scipy.special import erfc, erfcx, log_ndtr

def scores_kmnrl(theta, X, y):
    """
    Observation-wise scores for the classical normal linear regression model.

    theta = [beta, log(sigma)]

    Returns:
        G : n x (k+1) matrix of scores
    """

    theta = np.asarray(theta, dtype=float).copy()

    # Numerical stability clamp
    theta[-1] = np.clip(theta[-1], -40.0, 40.0)

    sigma2 = np.exp(2.0 * theta[-1])
    sigma2 = max(sigma2, 1e-12)

    beta = theta[:-1]

    e = y - X @ beta

    # Score wrt beta: n x k
    G_beta = X * (e / sigma2)[:, None]

    # Score wrt log(sigma): n x 1
    G_lsig = -1.0 + (e**2) / sigma2

    # Combine into n x (k+1) matrix
    G = np.column_stack((G_beta, G_lsig))

    return G


def scores_nex_a(theta, X, y):
    """
    Observation-wise scores for the normal-exponential SFA model.

    Parametrization A:
        theta = [beta, log(sv), log(su)]

    Returns:
        G : n x (k+2) matrix of scores
    """

    theta = np.asarray(theta, dtype=float).copy()

    k = X.shape[1]

    # Numerical stability clamp for log(sv), log(su)
    theta[-2:] = np.clip(theta[-2:], -20.0, 20.0)

    beta = theta[:k]
    s_v = np.exp(theta[k])
    s_u = np.exp(theta[k + 1])

    e = y - X @ beta

    a = -e / s_v - s_v / s_u

    # Stable logPhi
    logPhi = np.zeros_like(a)

    idxPhi = a < -10.0

    logPhi[idxPhi] = (np.log(0.5) + np.log(erfcx(-a[idxPhi] / np.sqrt(2.0))) - 0.5 * a[idxPhi] ** 2)

    logPhi[~idxPhi] = np.log(0.5 * erfc(-a[~idxPhi] / np.sqrt(2.0)))

    # Stable delta = phi(a) / Phi(a)
    logphi = -0.5 * a**2 - 0.5 * np.log(2.0 * np.pi)
    logdelta = logphi - logPhi

    delta = np.zeros_like(a)

    idx = a < -10.0

    delta[~idx] = np.exp(np.minimum(logdelta[~idx], np.log(1e14)))

    aa = a[idx]
    delta[idx] = -aa - 1.0 / aa + 2.0 / (aa**3)

    delta[~np.isfinite(delta)] = 1e14
    delta = np.minimum(delta, 1e14)

    # Score wrt beta: n x k
    G_beta = X * (-1.0 / s_u + delta / s_v)[:, None]

    # Helpful derivatives
    da_dsv = e / (s_v**2) - 1.0 / s_u
    da_dsu = s_v / (s_u**2)

    # Score wrt log(sv): n x 1
    G_log_sv = s_v * (s_v / (s_u**2) + delta * da_dsv)

    # Score wrt log(su): n x 1
    G_log_su = s_u * (-1.0 / s_u - (s_v**2) / (s_u**3) - e / (s_u**2) + delta * da_dsu)

    # Combine into n x (k+2)
    G = np.column_stack((G_beta, G_log_sv, G_log_su))

    return G


def scores_nhn_a(theta, X, y):
    """
    Score matrix for the Normal-Half-Normal SFA model.

    Parametrization A:
        theta = [beta, log(sv), log(su)]

    Output:
        G : ndarray, shape (n, k + 2)
            Matrix of individual score contributions.
    """

    theta = np.asarray(theta, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    k = X.shape[1]

    beta = theta[:k]
    s_v = np.exp(theta[k])
    s_u = np.exp(theta[k + 1])

    e = y - X @ beta

    s2 = s_v**2 + s_u**2
    s = np.sqrt(s2)
    lam = s_u / s_v

    a = -lam * e / s

    # log phi(a)
    logphi = -0.5 * a**2 - 0.5 * np.log(2.0 * np.pi)

    # Stable log Phi(a)
    logPhi = np.empty_like(a)

    idx = a < -10.0

    logPhi[idx] = (np.log(0.5) + np.log(erfcx(-a[idx] / np.sqrt(2.0))) - 0.5 * a[idx]**2)

    logPhi[~idx] = np.log(0.5 * erfc(-a[~idx] / np.sqrt(2.0)))

    # Stable inverse Mills ratio: delta = phi(a) / Phi(a)
    logdelta = logphi - logPhi
    delta = np.empty_like(a)

    delta[~idx] = np.exp(np.minimum(logdelta[~idx], np.log(1e12)))

    aa = a[idx]
    delta[idx] = -aa - 1.0 / aa + 2.0 / aa**3

    delta[~np.isfinite(delta)] = 1e12
    delta = np.minimum(delta, 1e14)

    # beta score
    G_beta = X * (e / s2 + delta * (lam / s))[:, None]

    # log(sv) score
    G_log_sv = (-(s_v**2 / s2) + e**2 * (s_v**2 / s2**2) + delta * (lam * e / s + lam * e * s_v**2 / s**3))

    # log(su) score
    G_log_su = (-(s_u**2 / s2) + e**2 * (s_u**2 / s2**2) + delta * (-lam * e / s + lam * e * s_u**2 / s**3))

    G = np.column_stack((G_beta, G_log_sv, G_log_su))

    return G


def scores_nexP_a(theta, X, y, n, T):
    """
    Score matrix for the panel normal-exponential SFA model.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]

    Parameters
    ----------
    theta : ndarray
        Parameter vector with length k + 2.
    X : ndarray
        Regressor matrix with shape (n*T, k).
    y : ndarray
        Dependent-variable vector with shape (n*T,).
    n : int
        Number of panel units.
    T : int
        Number of time observations per unit.

    Returns
    -------
    G : ndarray
        Unit-specific score matrix with shape (n, k + 2).
    """
    k = X.shape[1]

    beta = theta[:k]
    s_v = np.exp(theta[-2])
    s_u = np.exp(theta[-1])
    s2v = s_v**2

    residual = y - X @ beta

    # MATLAB-equivalent panel reshaping:
    # ee = reshape(e, T, n)
    residual_panel = residual.reshape((T, n), order="F")

    residual_mean = np.mean(residual_panel, axis=0)
    residual_ss = np.sum(residual_panel**2, axis=0)

    q = residual_ss / T

    c = s2v / (T * s_u)
    B = residual_mean + c

    d = q - B**2

    sqrt_T = np.sqrt(T)

    a = (-sqrt_T * residual_mean / s_v - s_v / (sqrt_T * s_u))

    # Stable log Phi(a)
    log_Phi = log_ndtr(a)

    # Inverse Mills ratio: phi(a) / Phi(a)
    log_phi = -0.5 * a**2 - 0.5 * np.log(2.0 * np.pi)
    mills_ratio = np.exp(log_phi - log_Phi)

    # MATLAB-equivalent reshaping:
    # XX = reshape(X, T, n, k)
    X_panel = X.reshape((T, n, k), order="F")

    X_mean = np.mean(X_panel, axis=0)

    EX_mean = np.mean(residual_panel[:, :, None] * X_panel, axis=0)

    G = np.zeros((n, k + 2))

    # Scores with respect to beta
    G[:, :k] = ((T / s2v) * (EX_mean - B[:, None] * X_mean) + (sqrt_T / s_v) * mills_ratio[:, None] * X_mean)

    # Score with respect to log(s_v)
    G[:, -2] = (-(T - 1) + (T / s2v) * d + 2.0 * B / s_u + mills_ratio * (sqrt_T * residual_mean / s_v - s_v / (sqrt_T * s_u)))

    # Score with respect to log(s_u)
    G[:, -1] = (-1.0 - B / s_u + mills_ratio * s_v / (sqrt_T * s_u))

    return G


def scores_nhnP_a(theta, X, y, n, T):
    """
    Score matrix for panel normal-half-normal SFA
    with time-invariant inefficiency.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]

    Returns
    -------
    G : ndarray
        Score matrix with shape (n, k + 2), where each row
        is the score contribution of one panel unit.
    """
    k = X.shape[1]

    beta = theta[:k]
    log_sv = theta[-2]
    log_su = theta[-1]

    s_v = np.exp(log_sv)
    s_u = np.exp(log_su)

    s2v = s_v**2
    s2u = s_u**2

    D = s2v + T * s2u

    residual = y - X @ beta

    residual_panel = residual.reshape((T, n), order="F")

    r = np.sum(residual_panel, axis=0)
    sse = np.sum(residual_panel**2, axis=0)

    a = s2u / D

    Q = sse - a * r**2

    z = -r * s_u / (s_v * np.sqrt(D))

    log_Phi = log_ndtr(z)

    log_phi = (-0.5 * z**2 - 0.5 * np.log(2.0 * np.pi))

    mills_ratio = np.exp(log_phi - log_Phi)

    X_panel = X.reshape((T, n, k), order="F")

    X_sum = np.sum(X_panel, axis=0)

    EX_sum = np.sum(residual_panel[:, :, None] * X_panel, axis=0)

    G = np.zeros((n, k + 2))

    c = s_u / (s_v * np.sqrt(D))

    G[:, :k] = (EX_sum - a * r[:, None] * X_sum) / s2v + mills_ratio[:, None] * c * X_sum

    # Score with respect to log(s_v)
    da_dlog_sv = -2.0 * s2u * s2v / D**2
    dQ_dlog_sv = -da_dlog_sv * r**2
    dz_dlog_sv = z * (-1.0 - s2v / D)

    G[:, -2] = (-(T - 1) - s2v / D - 0.5 * dQ_dlog_sv / s2v + Q / s2v + mills_ratio * dz_dlog_sv)

    # Score with respect to log(s_u)
    da_dlog_su = 2.0 * s2u * s2v / D**2
    dQ_dlog_su = -da_dlog_su * r**2
    dz_dlog_su = z * (1.0 - T * s2u / D)

    G[:, -1] = (-T * s2u / D - 0.5 * dQ_dlog_su / s2v + mills_ratio * dz_dlog_su)

    return G


def scores_re_a(theta, X, y, n, T):
    """
    Score matrix for the Gaussian panel random-effects model.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]

    Returns
    -------
    G : ndarray
        Matrix with shape (n, k + 2), where row i contains
        the score contribution of panel unit i.
    """
    k = X.shape[1]

    beta = theta[:k]
    s2v = np.exp(2.0 * theta[-2])
    s2u = np.exp(2.0 * theta[-1])

    d = s2v + T * s2u

    residual = y - X @ beta

    # MATLAB:
    # E = reshape(e, T, n)
    E = residual.reshape((T, n), order="F")

    # MATLAB:
    # X3 = reshape(X, T, n, k)
    # X3 = permute(X3, [1 3 2])
    X3 = X.reshape((T, n, k), order="F")
    X3 = np.transpose(X3, (0, 2, 1))  # T x k x n

    a = 1.0 / s2v
    c = s2u / (s2v * d)

    sum_e = np.sum(E, axis=0)
    sum_e2 = np.sum(E**2, axis=0)

    # Omega^{-1} e_i
    WE = a * E - c * sum_e[None, :]

    # Score with respect to beta
    G_beta = np.zeros((n, k))

    for i in range(n):
        Xi = X3[:, :, i]
        G_beta[i, :] = Xi.T @ WE[:, i]

    # Scores with respect to log(s2v) and log(s2u)
    dlogdet_dlogA = (T - 1) + s2v / d
    dlogdet_dlogB = T * s2u / d

    dq_dlogA_i = (-a * sum_e2 + c * (1.0 + s2v / d) * sum_e**2)

    dq_dlogB_i = (-c * (1.0 - T * s2u / d) * sum_e**2)

    score_logA_i = (-0.5 * dlogdet_dlogA -0.5 * dq_dlogA_i)

    score_logB_i = (-0.5 * dlogdet_dlogB -0.5 * dq_dlogB_i)

    # Chain rule:
    # log(s2v) = 2 log(s_v)
    # log(s2u) = 2 log(s_u)
    score_logsv_i = 2.0 * score_logA_i
    score_logsu_i = 2.0 * score_logB_i

    G = np.column_stack((G_beta, score_logsv_i, score_logsu_i))

    return G
