# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:10:42 2026

@author: makielak
"""

import numpy as np
from scipy.special import log_ndtr


def jondrow(model, X, y, bayes=0):
    """
    Calculate inefficiency and technical-efficiency estimates.

    Supported model names:
        'nex'  : pooled normal-exponential
        'nhn'  : pooled normal-half-normal
        'nexp' : panel normal-exponential
        'nhnp' : panel normal-half-normal
    """
    if bayes == 1:
        theta = model.bayes.theta_post
    else:
        theta = model.theta_ml
        
    if model.name == "nex":
        return eff_nex_a(theta, X, y)

    elif model.name == "nhn":
        return eff_nhn_a(theta, X, y)

    elif model.name == "nexp":
        return eff_nexP_a(theta, X, y, model.n, model.T)

    elif model.name == "nhnp":
        return eff_nhnP_a(theta,X, y, model.n, model.T)
    
    else:
        return 0.0, 1.0


def eff_nex_a(theta, X, y):
    """
    Jondrow inefficiency estimates for pooled normal-exponential SFA.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]
    """
    k = X.shape[1]

    beta = theta[:k]
    s_v = np.exp(theta[k])
    s_u = np.exp(theta[k + 1])

    residual = y - X @ beta

    a = -residual / s_v - s_v / s_u

    log_Phi = log_ndtr(a)
    log_phi = -0.5 * a**2 - 0.5 * np.log(2.0 * np.pi)

    delta = np.exp(log_phi - log_Phi)

    u_hat = (
        -residual
        - s_v**2 / s_u
        + s_v * delta
    )

    TE_hat = np.exp(-u_hat)

    return u_hat, TE_hat


def eff_nhn_a(theta, X, y):
    """
    Jondrow inefficiency estimates for pooled normal-half-normal SFA.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]
    """
    k = X.shape[1]

    beta = theta[:k]
    s_v = np.exp(theta[k])
    s_u = np.exp(theta[k + 1])

    residual = y - X @ beta

    sigma2 = s_u**2 + s_v**2

    mu_star = -residual * s_u**2 / sigma2
    sig_star = s_u * s_v / np.sqrt(sigma2)

    z = mu_star / sig_star

    log_Phi = log_ndtr(z)
    log_phi = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)

    delta = np.exp(log_phi - log_Phi)

    u_hat = mu_star + sig_star * delta

    # Numerical guard
    u_hat = np.maximum(u_hat, 0.0)

    TE_hat = np.exp(-u_hat)

    return u_hat, TE_hat


def eff_nexP_a(theta, X, y, n, T):
    """
    Efficiency estimates for panel normal-exponential SFA.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]

    Model:
        y_it = x_it' beta + v_it - u_i

        v_it ~ N(0, s_v^2)
        u_i  ~ Exponential(scale=s_u)

    Observations must be stacked as:
        unit 1 over all T periods,
        unit 2 over all T periods,
        ...

    Returns
    -------
    u_hat : ndarray, shape (n,)
        Conditional mean E(u_i | e_i).

    TE_hat : ndarray, shape (n,)
        Conditional mean E(exp(-u_i) | e_i).
    """
    k = len(theta) - 2

    beta = theta[:k]
    s_v = np.exp(theta[-2])
    s_u = np.exp(theta[-1])

    s2v = s_v**2

    residual = y - X @ beta

    # MATLAB equivalent:
    # ee = reshape(e, T, n)
    residual_panel = residual.reshape((T, n), order="F")

    # Sum of residuals over time for each unit
    r = np.sum(residual_panel, axis=0)

    sig2 = s2v / T
    sig = s_v / np.sqrt(T)

    mu = -r / T - s2v / (T * s_u)

    z = mu / sig

    log_Phi_z = log_ndtr(z)
    log_phi_z = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)

    mills_ratio = np.exp(log_phi_z - log_Phi_z)

    # Conditional mean inefficiency
    u_hat = mu + sig * mills_ratio

    # Exact truncated-normal moment:
    # E(exp(-u_i) | e_i)
    log_numerator_Phi = log_ndtr(
        (mu - sig2) / sig
    )

    TE_hat = np.exp(
        -mu
        + 0.5 * sig2
        + log_numerator_Phi
        - log_Phi_z
    )

    return u_hat, TE_hat


def eff_nhnP_a(theta, X, y, n, T):
    """
    Efficiency estimates for panel normal-half-normal SFA.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]

    Model:
        y_it = x_it' beta + v_it - u_i

        v_it ~ N(0, s_v^2)
        u_i  ~ |N(0, s_u^2)|

    Observations must be stacked as:
        unit 1 over all T periods,
        unit 2 over all T periods,
        ...

    Returns
    -------
    u_hat : ndarray, shape (n,)
        Conditional mean E(u_i | e_i).

    TE_hat : ndarray, shape (n,)
        Conditional mean E(exp(-u_i) | e_i).
    """
    k = len(theta) - 2

    beta = theta[:k]
    s_v = np.exp(theta[-2])
    s_u = np.exp(theta[-1])

    s2v = s_v**2
    s2u = s_u**2

    residual = y - X @ beta

    # MATLAB equivalent:
    # ee = reshape(e, T, n)
    residual_panel = residual.reshape((T, n), order="F")

    # Sum of residuals over time for each unit
    r = np.sum(residual_panel, axis=0)

    sig2 = 1.0 / (1.0 / s2u + T / s2v)
    sig = np.sqrt(sig2)

    mu = -sig2 * r / s2v

    z = mu / sig

    log_Phi_z = log_ndtr(z)
    log_phi_z = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)

    mills_ratio = np.exp(log_phi_z - log_Phi_z)

    # Jondrow estimator
    u_hat = mu + sig * mills_ratio

    # Exact truncated-normal moment:
    # E(exp(-u_i) | e_i)
    log_numerator_Phi = log_ndtr(
        (mu - sig2) / sig
    )

    TE_hat = np.exp(
        -mu
        + 0.5 * sig2
        + log_numerator_Phi
        - log_Phi_z
    )

    return u_hat, TE_hat