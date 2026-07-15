# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:55:17 2026

@author: makielak
"""

import numpy as np
from scipy.special import erfc, erfcx, expit


def lgl_kmnrl(theta, X, y):
    #migh comment this out too
    #theta = np.asarray(theta, dtype=float)
    #X = np.asarray(X, dtype=float)
    #y = np.asarray(y, dtype=float).reshape(-1)

    beta = theta[:-1]
    lsv = theta[-1]

    s2v = np.exp(2.0 * lsv)
    n = y.size

    e = y - X @ beta

    L = ( -0.5 * n * np.log(2.0 * np.pi) - n * lsv - (e @ e) / (2.0 * s2v) )

    g_beta = X.T @ e / s2v
    g_lsv = -n + (e @ e) / s2v

    grad = np.r_[g_beta, g_lsv]

    return L, grad

##-- normal-exponential

def nlgl_nex_b(theta, X, y):
    """
    Negative log-likelihood for the normal-exponential SFA model.

    Parametrization B:
        theta = [beta, log_sigma2, eta]

    where:
        sigma2 = exp(log_sigma2)
        gamma = logistic(eta)
        s_u = sqrt(gamma * sigma2)
        s_v = sqrt((1 - gamma) * sigma2)

    Returns:
        nL    : negative log-likelihood
        ngrad : gradient of negative log-likelihood
    """

    #theta = np.asarray(theta, dtype=float)

    k = X.shape[1]

    # PARAMETERS
    beta = theta[:k]
    log_sigma2 = theta[k]
    eta = theta[k + 1]

    # TRANSFORMS
    sigma2 = np.exp(log_sigma2)

    # Stable logistic transformation
    gamma = expit(eta)
    gamma = np.clip(gamma, 1e-12, 1.0 - 1e-12)
    
    s_u = np.sqrt(gamma * sigma2)
    s_v = np.sqrt((1.0 - gamma) * sigma2)

    # Numerical stability
    s_v = max(s_v, 1e-10)
    s_u = max(s_u, 1e-10)

    # RESIDUALS
    e = y - X @ beta

    # CORE TERMS
    a = -e / s_v - s_v / s_u

    logPhi = np.zeros_like(a)

    idxPhi = a < -10.0
    logPhi[idxPhi] = ( np.log(0.5) + np.log(erfcx(-a[idxPhi] / np.sqrt(2.0))) - 0.5 * a[idxPhi] ** 2 )
    logPhi[~idxPhi] = np.log( 0.5 * erfc(-a[~idxPhi] / np.sqrt(2.0)) )

    logphi = -0.5 * a**2 - 0.5 * np.log(2.0 * np.pi)

    # delta = phi(a) / Phi(a), computed stably
    logdelta = logphi - logPhi

    delta = np.zeros_like(a)
    idx = a < -10.0
    delta[~idx] = np.exp(np.minimum(logdelta[~idx], np.log(1e12)))

    # Asymptotic approximation for very negative a
    aa = a[idx]
    delta[idx] = -aa - 1.0 / aa + 2.0 / (aa**3)

    # Safety
    delta[~np.isfinite(delta)] = 1e14
    delta = np.minimum(delta, 1e14)

    # LOG-LIKELIHOOD
    loglik_i = ( -np.log(s_u) + 0.5 * (s_v**2) / (s_u**2) + e / s_u + logPhi)

    nL = -np.sum(loglik_i)

    # GRADIENT

    # beta
    g_beta = X.T @ (-1.0 / s_u + delta / s_v)

    # Derivatives wrt s_v and s_u
    da_dsv = e / (s_v**2) - 1.0 / s_u
    da_dsu = s_v / (s_u**2)

    d_sv = s_v / (s_u**2) + delta * da_dsv

    d_su = ( -1.0 / s_u - e / (s_u**2) - (s_v**2) / (s_u**3) + delta * da_dsu )

    # Chain rule to sigma2 and gamma
    dsu_dsigma2 = gamma / (2.0 * s_u)
    dsv_dsigma2 = (1.0 - gamma) / (2.0 * s_v)

    dsu_dgamma = sigma2 / (2.0 * s_u)
    dsv_dgamma = -sigma2 / (2.0 * s_v)

    # log_sigma2
    g_sigma2 = np.sum(d_su * dsu_dsigma2 + d_sv * dsv_dsigma2)
    g_log_sigma2 = g_sigma2 * sigma2

    # eta / logit-gamma
    g_gamma = np.sum(d_su * dsu_dgamma + d_sv * dsv_dgamma)
    g_logit_gamma = g_gamma * gamma * (1.0 - gamma)

    # Negative gradient because nL = -L
    ngrad = -np.r_[g_beta, g_log_sigma2, g_logit_gamma]

    return nL, ngrad



def lgl_nex_a(theta, X, y):
    """
    Log-likelihood for the normal-exponential SFA model.

    Parametrization A:
        theta = [beta, log(sv), log(su)]

    Returns:
        L : log-likelihood value
    """

    #theta = np.asarray(theta, dtype=float)

    n = y.size

    # Numerical stability clamp for log(sv), log(su)
    theta[-2:] = np.clip(theta[-2:], -20.0, 20.0)

    s_v = np.exp(theta[-2])
    s_u = np.exp(theta[-1])

    beta = theta[:-2]

    e = y - X @ beta

    a = -e / s_v - s_v / s_u

    logPhi = np.zeros_like(a)

    idx = a < -10.0

    logPhi[idx] = ( np.log(0.5) + np.log(erfcx(-a[idx] / np.sqrt(2.0))) - 0.5 * a[idx] ** 2 )

    logPhi[~idx] = np.log( 0.5 * erfc(-a[~idx] / np.sqrt(2.0)) )

    L = ( -n * np.log(s_u) + 0.5 * n * (s_v**2) / (s_u**2) + np.sum(e) / s_u + np.sum(logPhi) )

    return L

##-- normal-half-normal part

def nlgl_nhn_b(theta, X, y):
    """
    Negative log-likelihood for the Normal-Half-Normal SFA model.

    Parametrization B:
        theta = [beta, log_sigma2, eta]

    where:
        sigma2 = exp(log_sigma2)
        gamma  = logistic(eta)
        s_u    = sqrt(gamma * sigma2)
        s_v    = sqrt((1 - gamma) * sigma2)

    Parameters
    ----------
    theta : ndarray
        Parameter vector [beta, log_sigma2, eta].
    X : ndarray
        Regressor matrix, shape (n, k).
    y : ndarray
        Dependent variable, shape (n,).
    grad : bool, default False
        If True, also returns the gradient.

    Returns
    -------
    nL : float
        Negative log-likelihood.
    ngrad : ndarray, optional
        Gradient of negative log-likelihood.
    """

    #theta = np.asarray(theta, dtype=float)
    #X = np.asarray(X, dtype=float)
    #y = np.asarray(y, dtype=float).reshape(-1)

    k = X.shape[1]

    # PARAMETERS
    beta = theta[:k]
    log_sigma2 = theta[k]
    eta = theta[k + 1]

    sigma2 = np.exp(log_sigma2)

    # Stable logistic transformation
    if eta >= 0:
        gamma = 1.0 / (1.0 + np.exp(-eta))
    else:
        eeta = np.exp(eta)
        gamma = eeta / (1.0 + eeta)

    gamma = np.clip(gamma, 1e-12, 1.0 - 1e-12)

    s_u = np.sqrt(gamma * sigma2)
    s_v = np.sqrt((1.0 - gamma) * sigma2)

    sigma = np.sqrt(sigma2)
    lam = s_u / s_v

    # RESIDUALS
    e = y - X @ beta
    a = e / sigma
    w = -lam * a

    # LOG COMPONENTS
    logphi_a = -0.5 * a**2 - 0.5 * np.log(2.0 * np.pi)

    # Stable log Phi(w)
    logPhi_w = np.empty_like(w)

    idxPhi = w < -10
    logPhi_w[idxPhi] = (
        np.log(0.5)
        + np.log(erfcx(-w[idxPhi] / np.sqrt(2.0)))
        - 0.5 * w[idxPhi] ** 2
    )

    logPhi_w[~idxPhi] = np.log(
        0.5 * erfc(-w[~idxPhi] / np.sqrt(2.0))
    )

    # LOG-LIKELIHOOD
    loglik = np.sum(
        -np.log(sigma)
        + np.log(2.0)
        + logphi_a
        + logPhi_w
    )

    nL = -loglik

    # GRADIENT

    logphi_w = -0.5 * w**2 - 0.5 * np.log(2.0 * np.pi)
    logdelta = logphi_w - logPhi_w

    delta = np.empty_like(w)

    idx = w < -10

    delta[~idx] = np.exp(np.minimum(logdelta[~idx], np.log(1e12)))

    ww = w[idx]
    delta[idx] = -ww - 1.0 / ww + 2.0 / (ww**3)

    delta[~np.isfinite(delta)] = 1e12
    delta = np.minimum(delta, 1e14)

    # beta
    g_beta = X.T @ (a / sigma + delta * (lam / sigma))

    # derivatives wrt sigma and lambda
    d_sigma = -1.0 / sigma + a**2 / sigma + delta * (lam * a / sigma)

    d_lambda = -delta * a

    # chain rule to sigma2 and gamma

    # sigma = sqrt(sigma2)
    d_sigma2 = d_sigma * (1.0 / (2.0 * sigma))

    # lambda = sqrt(gamma / (1 - gamma))
    d_lambda_dgamma = 1.0 / (2.0 * lam * (1.0 - gamma) ** 2)

    g_sigma2 = d_sigma2
    g_gamma = d_lambda * d_lambda_dgamma

    # log transforms
    g_log_sigma2 = np.sum(g_sigma2) * sigma2
    g_logit_gamma = np.sum(g_gamma) * gamma * (1.0 - gamma)

    ngrad = -np.r_[g_beta, g_log_sigma2, g_logit_gamma]

    return nL, ngrad


def lgl_nhn_a(theta, X, y):
    """
    Log-likelihood for the Normal-Half-Normal SFA model.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]

    where:
        s_v = exp(theta[-2])
        s_u = exp(theta[-1])

    Parameters
    ----------
    theta : ndarray
        Parameter vector [beta, log(s_v), log(s_u)].
    X : ndarray
        Regressor matrix, shape (n, k).
    y : ndarray
        Dependent variable, shape (n,).

    Returns
    -------
    L : float
        Log-likelihood value.
    """

    #theta = np.asarray(theta, dtype=float)
    #X = np.asarray(X, dtype=float)
    #y = np.asarray(y, dtype=float).reshape(-1)

    n = y.size

    # Clamp log standard deviations for numerical stability
    theta[-2:] = np.clip(theta[-2:], -20.0, 20.0)

    s_v = np.exp(theta[-2])
    s_u = np.exp(theta[-1])

    s = np.sqrt(s_v**2 + s_u**2)
    lbd = s_u / s_v

    beta = theta[:-2]

    e = y - X @ beta
    a = -e * lbd / s

    # Stable log Phi(a)
    logPhi = np.empty_like(a)

    idx = a < -10

    logPhi[idx] = (
        np.log(0.5)
        + np.log(erfcx(-a[idx] / np.sqrt(2.0)))
        - 0.5 * a[idx] ** 2
    )

    logPhi[~idx] = np.log(
        0.5 * erfc(-a[~idx] / np.sqrt(2.0))
    )

    L = (
        -0.5 * n * np.log(np.pi / 2.0)
        - n * np.log(s)
        + np.sum(logPhi)
        - 0.5 * np.dot(e, e) / s**2
    )

    return L

##-- panel normal-exponential

from scipy.special import log_ndtr

def nlgl_nexP_b(theta, X, y, n, T):
    """
    Negative log-likelihood for the panel normal-exponential SFA model.

    Parametrization B:
        theta = [beta, log(sigma^2), logit(gamma)]

    Parameters
    ----------
    theta : array_like
        Parameter vector [beta, log_sigma2, logit_gamma].
    X : ndarray
        Regressor matrix with shape (n*T, k).
    y : ndarray
        Dependent variable with length n*T.
    n : int
        Number of cross-sectional units.
    T : int
        Number of observations per unit.
    return_gradient : bool, default=False
        Return the analytical gradient when True.

    Returns
    -------
    nL : float
        Negative log-likelihood.
    gradient : ndarray, optional
        Gradient of the negative log-likelihood.
    """
    #theta = np.asarray(theta, dtype=float)
    #X = np.asarray(X, dtype=float)
    #y = np.asarray(y, dtype=float).reshape(-1)

    k = X.shape[1]

    beta = theta[:k]
    log_sigma2 = theta[-2]
    eta = theta[-1]

    sigma2 = np.exp(log_sigma2)

    # Stable logistic transformation
    gamma = np.clip(
        expit(eta),
        1.0e-12,
        1.0 - 1.0e-12,
    )

    sigma2_u = gamma * sigma2
    sigma2_v = (1.0 - gamma) * sigma2

    sigma_u = np.sqrt(sigma2_u)
    sigma_v = np.sqrt(sigma2_v)

    residual = y - X @ beta

    # MATLAB equivalent:
    # ee = reshape(e, T, n)
    residual_panel = residual.reshape((T, n), order="F")

    residual_mean = np.mean(residual_panel, axis=0)
    residual_ss = np.sum(residual_panel**2, axis=0)

    # MATLAB equivalent:
    # XX = reshape(X, T, n, k)
    X_panel = X.reshape((T, n, k), order="F")

    X_mean = np.mean(X_panel, axis=0)

    # Unit-specific means of e_it * x_itj
    EX_mean = np.mean(
        residual_panel[:, :, None] * X_panel,
        axis=0,
    )

    c = sigma2_v / (T * sigma_u)
    B = residual_mean + c

    sqrt_T = np.sqrt(T)

    a = (
        -sqrt_T * residual_mean / sigma_v
        - sigma_v / (sqrt_T * sigma_u)
    )

    # Stable log(Phi(a))
    log_Phi = log_ndtr(a)

    L1 = (
        -n * np.log(sigma_u)
        - 0.5 * n * (T - 1)
        * np.log(2.0 * np.pi * sigma2_v)
        - 0.5 * n * np.log(T)
    )

    L2 = (
        -T
        / (2.0 * sigma2_v)
        * np.sum(residual_ss / T - B**2)
    )

    L3 = np.sum(log_Phi)

    nL = float(-(L1 + L2 + L3))

    # Stable inverse Mills ratio: phi(a) / Phi(a)
    log_phi = (
        -0.5 * a**2
        - 0.5 * np.log(2.0 * np.pi)
    )

    log_delta = log_phi - log_Phi

    delta = np.empty_like(a)

    tail = a < -10.0

    delta[~tail] = np.exp(
        np.minimum(
            log_delta[~tail],
            np.log(1.0e12),
        )
    )

    aa = a[tail]

    delta[tail] = (
        -aa
        - 1.0 / aa
        + 2.0 / aa**3
    )

    delta[~np.isfinite(delta)] = 1.0e12
    delta = np.minimum(delta, 1.0e14)

    # Gradient with respect to beta
    grad_beta = np.sum(
        (T / sigma2_v)
        * (
            EX_mean
            - B[:, None] * X_mean
        )
        + (sqrt_T / sigma_v)
        * delta[:, None]
        * X_mean,
        axis=0,
    )

    d = residual_ss / T - B**2

    dL_dsigma2_v = np.sum(
        T * d / (2.0 * sigma2_v**2)
        + B / (sigma2_v * sigma_u)
    )

    dL_dsigma_v = (
        2.0 * sigma_v * dL_dsigma2_v
        - n * (T - 1) / sigma_v
        + np.sum(
            delta
            * (
                sqrt_T
                * residual_mean
                / sigma_v**2
                - 1.0
                / (sqrt_T * sigma_u)
            )
        )
    )

    dL_dsigma_u = (
        -n / sigma_u
        - np.sum(B / sigma_u**2)
        + np.sum(
            delta
            * sigma_v
            / (sqrt_T * sigma_u**2)
        )
    )

    # Chain rule for log(sigma^2)
    dsigma_v_dlog_sigma2 = 0.5 * sigma_v
    dsigma_u_dlog_sigma2 = 0.5 * sigma_u

    # Chain rule for eta = logit(gamma)
    dsigma_v_deta = -0.5 * sigma_v * gamma
    dsigma_u_deta = (
        0.5
        * sigma_u
        * (1.0 - gamma)
    )

    grad_log_sigma2 = (
        dL_dsigma_v
        * dsigma_v_dlog_sigma2
        + dL_dsigma_u
        * dsigma_u_dlog_sigma2
    )

    grad_eta = (
        dL_dsigma_v * dsigma_v_deta
        + dL_dsigma_u * dsigma_u_deta
    )

    gradient = -np.concatenate(
        (
            grad_beta,
            np.array([
                grad_log_sigma2,
                grad_eta,
            ]),
        )
    )

    return nL, gradient

def lgl_nexP_a(theta, X, y, n, T):
    """
    Log-likelihood for the panel normal-exponential SFA model.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]

    The inefficiency term is persistent within each panel unit.

    Parameters
    ----------
    theta : array_like
        Parameter vector containing regression coefficients followed by
        log(s_v) and log(s_u).
    X : array_like
        Regressor matrix with shape (n*T, k).
    y : array_like
        Dependent-variable vector with length n*T.
    n : int
        Number of panel units.
    T : int
        Number of time observations per panel unit.

    Returns
    -------
    float
        Log-likelihood value.
    """
    #theta = np.asarray(theta, dtype=float).reshape(-1)
    #X = np.asarray(X, dtype=float)
    #y = np.asarray(y, dtype=float).reshape(-1)

    k = X.shape[1]

    if theta.size != k + 2:
        raise ValueError(
            f"theta must contain {k + 2} elements: "
            f"{k} beta coefficients, log(s_v), and log(s_u)."
        )

    if X.shape[0] != n * T:
        raise ValueError(
            f"X must have n*T = {n * T} rows, but has {X.shape[0]}."
        )

    if y.size != n * T:
        raise ValueError(
            f"y must contain n*T = {n * T} observations, "
            f"but contains {y.size}."
        )

    beta = theta[:k]

    # Match the MATLAB stability restriction.
    log_sv = np.clip(theta[-2], -20.0, 20.0)
    log_su = np.clip(theta[-1], -20.0, 20.0)

    s_v = np.exp(log_sv)
    s_u = np.exp(log_su)
    s2v = s_v**2

    residual = y - X @ beta

    # MATLAB:
    # ee = reshape(e, T, n)
    #
    # order="F" reproduces MATLAB's column-major reshape.
    residual_panel = residual.reshape((T, n), order="F")

    residual_mean = np.mean(residual_panel, axis=0)
    residual_ss = np.sum(residual_panel**2, axis=0)

    arg1 = (
        -n * np.log(s_u)
        - 0.5 * n * (T - 1) * np.log(2.0 * np.pi * s2v)
        - 0.5 * n * np.log(T)
    )

    adjusted_mean = residual_mean + s2v / (T * s_u)

    arg2 = (
        -T
        / (2.0 * s2v)
        * np.sum(residual_ss / T - adjusted_mean**2)
    )

    a = (
        -np.sqrt(T) * residual_mean / s_v
        - s_v / (np.sqrt(T) * s_u)
    )

    # Numerically stable log(Phi(a)).
    arg3 = np.sum(log_ndtr(a))

    return float(arg1 + arg2 + arg3)


##-- panel normal-half-normal

def nlgl_nhnP_b(theta, X, y, n, T):
    """
    Negative log-likelihood for panel normal-half-normal SFA.

    Parametrization B:
        theta = [beta, log(sigma2), eta]

        sigma2 = s2u + s2v
        gamma  = s2u / (s2u + s2v)
        eta    = logit(gamma)

    Returns
    -------
    L : float
        Negative log-likelihood.

    grad : ndarray
        Analytical gradient of the negative log-likelihood.
    """
    k = X.shape[1]

    beta = theta[:k]
    log_sigma2 = theta[-2]
    eta = theta[-1]

    sigma2 = np.exp(log_sigma2)

    gamma = np.clip(
        expit(eta),
        1e-12,
        1.0 - 1e-12,
    )

    s2u = gamma * sigma2
    s2v = (1.0 - gamma) * sigma2

    s_u = np.sqrt(s2u)
    s_v = np.sqrt(s2v)

    D = s2v + T * s2u

    residual = y - X @ beta

    # MATLAB:
    # ee = reshape(e, T, n)
    residual_panel = residual.reshape((T, n), order="F")

    r = np.sum(residual_panel, axis=0)
    sse = np.sum(residual_panel**2, axis=0)

    a = s2u / D

    Q = sse - a * r**2

    z = -r * s_u / (s_v * np.sqrt(D))

    log_Phi = log_ndtr(z)

    L1 = (
        n * np.log(2.0)
        - 0.5 * n * T * np.log(2.0 * np.pi)
        - 0.5 * n * (T - 1) * np.log(s2v)
        - 0.5 * n * np.log(D)
    )

    L2 = -0.5 * np.sum(Q / s2v)

    L3 = np.sum(log_Phi)

    L = float(-(L1 + L2 + L3))

    # Stable inverse Mills ratio: phi(z) / Phi(z)
    log_phi = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)
    log_delta = log_phi - log_Phi

    delta = np.empty_like(z)

    tail = z < -10.0

    delta[~tail] = np.exp(
        np.minimum(
            log_delta[~tail],
            np.log(1e12),
        )
    )

    zz = z[tail]

    delta[tail] = (
        -zz
        - 1.0 / zz
        + 2.0 / zz**3
    )

    delta[~np.isfinite(delta)] = 1e12
    delta = np.minimum(delta, 1e14)

    # MATLAB:
    # XX = reshape(X, T, n, k)
    X_panel = X.reshape((T, n, k), order="F")

    X_sum = np.sum(X_panel, axis=0)

    EX_sum = np.sum(
        residual_panel[:, :, None] * X_panel,
        axis=0,
    )

    c = s_u / (s_v * np.sqrt(D))

    grad_beta = np.sum(
        (
            EX_sum
            - a * r[:, None] * X_sum
        )
        / s2v
        + delta[:, None] * c * X_sum,
        axis=0,
    )

    # Derivatives with respect to s2v and s2u
    dz_ds2v = z * (
        -0.5 / s2v
        - 0.5 / D
    )

    dz_ds2u = z * (
        0.5 / s2u
        - 0.5 * T / D
    )

    dL_ds2v = (
        -0.5 * (T - 1) / s2v
        - 0.5 / D
        - 0.5 * s2u * r**2 / (D**2 * s2v)
        + 0.5 * Q / s2v**2
        + delta * dz_ds2v
    )

    dL_ds2u = (
        -0.5 * T / D
        + 0.5 * r**2 / D**2
        + delta * dz_ds2u
    )

    # Chain rule for log(sigma2)
    ds2v_dlog_sigma2 = s2v
    ds2u_dlog_sigma2 = s2u

    # Chain rule for eta = logit(gamma)
    ds2v_deta = -gamma * s2v
    ds2u_deta = (1.0 - gamma) * s2u

    grad_log_sigma2 = np.sum(
        dL_ds2v * ds2v_dlog_sigma2
        + dL_ds2u * ds2u_dlog_sigma2
    )

    grad_eta = np.sum(
        dL_ds2v * ds2v_deta
        + dL_ds2u * ds2u_deta
    )

    grad = -np.concatenate(
        (
            grad_beta,
            np.array([
                grad_log_sigma2,
                grad_eta,
            ]),
        )
    )

    return L, grad

def lgl_nhnP_a(theta, X, y, n, T):
    """
    Log-likelihood for panel normal-half-normal SFA
    with persistent inefficiency.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]
    """
    k = X.shape[1]

    beta = theta[:k]

    log_sv = np.clip(theta[-2], -20.0, 20.0)
    log_su = np.clip(theta[-1], -20.0, 20.0)

    s_v = np.exp(log_sv)
    s_u = np.exp(log_su)

    s2v = s_v**2
    s2u = s_u**2

    sigs2 = s2v + T * s2u

    residual = y - X @ beta

    # MATLAB equivalent:
    # ee = reshape(e, T, n)
    residual_panel = residual.reshape((T, n), order="F")

    residual_mean = np.mean(residual_panel, axis=0)
    residual_ss = np.sum(residual_panel**2, axis=0)
    residual_sum = np.sum(residual_panel, axis=0)

    arg1 = (
        n * np.log(2.0)
        - 0.5 * n * T * np.log(2.0 * np.pi)
        - 0.5 * n * (T - 1) * np.log(s2v)
        - 0.5 * n * np.log(sigs2)
    )

    arg2 = (
        -0.5
        / s2v
        * np.sum(
            residual_ss
            - (s2u / sigs2) * residual_sum**2
        )
    )

    a = (
        -residual_mean
        * T
        * s_u
        / (s_v * np.sqrt(sigs2))
    )

    arg3 = np.sum(log_ndtr(a))

    return float(arg1 + arg2 + arg3)

##-- panel random effects model


def nlgl_re_b(theta, X, y, n, T):
    """
    Negative log-likelihood for a Gaussian panel random-effects model.

    Parametrization B:
        theta = [beta, log(sigma2), logit(gamma)]

    where:
        sigma2 = s_v^2 + s_u^2
        gamma  = s_u^2 / sigma2

        s_v^2 = (1 - gamma) * sigma2
        s_u^2 = gamma * sigma2

    Returns
    -------
    nL : float
        Negative log-likelihood.

    ng : ndarray
        Gradient of the negative log-likelihood.
    """
    k = X.shape[1]

    beta = theta[:k]

    # Numerical stability
    log_sigma2 = np.clip(theta[k], -30.0, 30.0)
    sigma2 = np.exp(log_sigma2)

    logit_gamma = theta[k + 1]

    gamma = np.clip(
        expit(logit_gamma),
        1e-12,
        1.0 - 1e-12,
    )

    A = (1.0 - gamma) * sigma2   # s_v^2
    B = gamma * sigma2           # s_u^2

    residual = y - X @ beta

    # MATLAB equivalent:
    # E = reshape(e, T, n)
    E = residual.reshape((T, n), order="F")

    d = A + T * B
    a = 1.0 / A
    c = B / (A * d)

    sum_e2 = np.sum(E**2, axis=0)
    sum_e = np.sum(E, axis=0)

    quad_i = a * sum_e2 - c * sum_e**2
    quad = np.sum(quad_i)

    logdet_omega = (T - 1) * np.log(A) + np.log(d)

    log_likelihood = (
        -0.5 * n * T * np.log(2.0 * np.pi)
        -0.5 * n * logdet_omega
        -0.5 * quad
    )

    nL = float(-log_likelihood)

    # Gradient with respect to beta
    WE = a * E - c * sum_e[None, :]

    # MATLAB WE(:) uses column-major ordering.
    score_beta = X.T @ WE.reshape(-1, order="F")

    # Scores with respect to log(A) and log(B)
    dlogdet_dlogA = (T - 1) + A / d
    dlogdet_dlogB = T * B / d

    dq_dlogA_i = (
        -a * sum_e2
        + c * (1.0 + A / d) * sum_e**2
    )

    dq_dlogB_i = (
        -c
        * (1.0 - T * B / d)
        * sum_e**2
    )

    score_logA = (
        -0.5 * n * dlogdet_dlogA
        -0.5 * np.sum(dq_dlogA_i)
    )

    score_logB = (
        -0.5 * n * dlogdet_dlogB
        -0.5 * np.sum(dq_dlogB_i)
    )

    # Chain rule to parametrization B
    score_log_sigma2 = score_logA + score_logB

    score_logit_gamma = (
        -gamma * score_logA
        + (1.0 - gamma) * score_logB
    )

    ng = -np.concatenate(
        (
            score_beta,
            np.array([
                score_log_sigma2,
                score_logit_gamma,
            ]),
        )
    )

    return nL, ng


def lgl_re_a(theta, X, y, n, T):
    """
    Log-likelihood for the Gaussian panel random-effects model.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]

    Observations are stacked as:
        unit 1 over all T periods,
        unit 2 over all T periods,
        ...
    """
    k = X.shape[1]

    beta = theta[:k]

    log_sv = np.clip(theta[-2], -20.0, 20.0)
    log_su = np.clip(theta[-1], -20.0, 20.0)

    s2v = np.exp(2.0 * log_sv)
    s2u = np.exp(2.0 * log_su)

    sig2 = s2v + T * s2u

    residual = y - X @ beta

    # MATLAB equivalent:
    # ee = reshape(e, T, n)
    residual_panel = residual.reshape((T, n), order="F")

    sse = np.sum(residual_panel**2, axis=0)
    esum = np.sum(residual_panel, axis=0)

    arg1 = (
        -0.5 * n * T * np.log(2.0 * np.pi)
        -0.5 * n * (T - 1) * np.log(s2v)
        -0.5 * n * np.log(sig2)
    )

    arg2 = (
        -0.5 / s2v
        * np.sum(
            sse
            - (s2u / sig2) * esum**2
        )
    )

    return float(arg1 + arg2)
