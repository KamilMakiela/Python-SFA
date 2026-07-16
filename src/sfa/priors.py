# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:17:58 2026

@author: makielak
"""

import numpy as np
from scipy.special import gammaln, expit

def prior_hes_kmnrl(theta):
    """
    Hessian of the log prior for the classical normal linear regression model.

    theta = [beta, log(sigma)]

    beta ~ N(0, 100 I)
    sigma^2 ~ IG(0.00005, 0.00005)
    """

    theta = np.asarray(theta, dtype=float)

    k = len(theta) - 1
    b = 0.00005

    H_prior = np.zeros((k + 1, k + 1))

    # beta prior: beta ~ N(0, 100 I)
    H_prior[:k, :k] = -np.eye(k) / 100.0

    # log(sigma) prior induced by sigma^2 ~ IG(0.00005, 0.00005)
    H_prior[k, k] = -4.0 * b * np.exp(-2.0 * theta[-1])

    return H_prior


def lg_pr_kmnrl(theta):
    """
    Log prior for the classical normal linear regression model.

    Parametrization:
        theta = [beta, log(sv)]

    Priors:
        beta ~ N(0, 100 I)
        sv^2 ~ IG(a, b), with a = b = 0.00005

    Returns:
        lgp  : log prior
        grad : gradient of log prior
    """

    theta = np.asarray(theta, dtype=float)

    k = len(theta) - 1

    beta = theta[:-1]
    lsv = theta[-1]

    a = 0.00005
    b = 0.00005

    # log prior for beta: beta ~ N(0, 100 I)
    lgp_b = ( -0.5 * k * np.log(2.0 * np.pi) -0.5 * k * np.log(100.0) -0.5 * np.sum(beta**2) / 100.0 )

    # log prior for sv^2, with transformation sv^2 = exp(2*lsv)
    s2v = np.exp(2.0 * lsv)

    # inverse-gamma log density:
    # p(x) = b^a / Gamma(a) * x^(-a-1) * exp(-b/x)
    log_ig = ( a * np.log(b) - gammaln(a) - (a + 1.0) * np.log(s2v) - b / s2v )

    # Jacobian: d(sv^2) / d(log sv) = 2 * sv^2
    log_jac = np.log(2.0) + np.log(s2v)

    lgp_sv = log_ig + log_jac

    lgp = lgp_b + lgp_sv

    # Gradient wrt beta
    g_beta = -beta / 100.0

    # Gradient wrt log(sv)
    g_lsv = -2.0 * a + 2.0 * b * np.exp(-2.0 * lsv)

    grad = np.r_[g_beta, g_lsv]

    return lgp, grad


##-- normal-exponential

def lg_pr_nex_b(theta):
    """
    Log prior and analytic gradient for the normal-exponential SFA model.

    Parametrization B:
        theta = [beta, log_sigma2, eta]

    where:
        gamma = logistic(eta)
        sigma2 = exp(log_sigma2)
        s_u = sqrt(gamma * sigma2)
        s2v = (1 - gamma) * sigma2

    Returns:
        lgp : scalar log prior
        g   : gradient of lgp with respect to theta
    """

    theta = np.asarray(theta, dtype=float)

    k = len(theta) - 2

    beta = theta[:k]
    log_sigma2 = theta[-2]
    eta = theta[-1]

    # Hyperparameters
    a_v = 0.00005
    b_v = 0.00005

    a_u = 1.0
    b_u = 0.2877

    # Transformations
    sigma2 = np.exp(log_sigma2)

    # Stable logistic
    gamma = expit(eta)

    s_u = np.sqrt(gamma * sigma2)
    s2v = (1.0 - gamma) * sigma2

    # Log prior for beta: beta ~ N(0, 100 I)
    lgp_b = ( -0.5 * k * np.log(2.0 * np.pi) -0.5 * k * np.log(100.0) -0.5 * (beta @ beta) / 100.0 )

    # sigma_v^2 ~ InvGamma(a_v, b_v)
    lgp_sv = ( a_v * np.log(b_v) - gammaln(a_v) - (a_v + 1.0) * np.log(s2v) - b_v / s2v )

    # sigma_u ~ InvGamma(a_u, b_u)
    lgp_su = ( a_u * np.log(b_u) - gammaln(a_u) - (a_u + 1.0) * np.log(s_u) - b_u / s_u )

    # Jacobian
    logJ = np.log(0.5) + np.log(s2v) + np.log(s_u)

    lgp = lgp_b + lgp_sv + lgp_su + logJ

    # Gradient
    g = np.zeros_like(theta)

    # d/d beta
    g[:k] = -beta / 100.0

    # d/d log_sigma2
    d_lgp_sv_dlogs2 = -(a_v + 1.0) + b_v / s2v

    d_lgp_su_dlogs2 = ( -0.5 * (a_u + 1.0) + 0.5 * b_u / s_u )

    d_lgJ_dlogs2 = 1.5

    g[-2] = d_lgp_sv_dlogs2 + d_lgp_su_dlogs2 + d_lgJ_dlogs2

    # d/d eta
    d_lgp_sv_deta = gamma * (a_v + 1.0) - gamma * b_v / s2v

    d_lgp_su_deta = ( 0.5 * (1.0 - gamma) * (-(a_u + 1.0) + b_u / s_u) )

    d_lgJ_deta = 0.5 - 1.5 * gamma

    g[-1] = d_lgp_sv_deta + d_lgp_su_deta + d_lgJ_deta

    return lgp, g


def lg_pr_nex_a(theta):
    """
    Log prior for the normal-exponential SFA model.

    Parametrization A:
        theta = [beta, log(sv), log(su)]

    Priors:
        beta ~ N(0, 100 I)
        sv^2 ~ IG(av, bv)
        su ~ IG(au, bu)

    Returns:
        lgp : log prior value
    """

    theta = np.asarray(theta, dtype=float)

    k = len(theta) - 2

    beta = theta[:-2]
    lv = theta[-2]   # log(s_v)
    lu = theta[-1]   # log(s_u)

    av = 0.00005
    bv = 0.00005

    au = 1.0
    bu = 0.2877

    # beta ~ N(0, 100 I)
    lgp_b = ( -0.5 * k * np.log(2.0 * np.pi * 100.0) -0.5 * (beta @ beta) / 100.0 )

    # sv^2 prior + Jacobian, simplified
    lgp_sv = ( av * np.log(bv) - gammaln(av) - 2.0 * av * lv - bv * np.exp(-2.0 * lv) + np.log(2.0) )

    # su prior + Jacobian, simplified
    lgp_su = ( au * np.log(bu) - gammaln(au) - au * lu - bu * np.exp(-lu) )

    lgp = lgp_b + lgp_sv + lgp_su

    return lgp


def prior_hes_nex_a(theta):
    """
    Hessian of the log prior for the normal-exponential SFA model.

    Parametrization A:
        theta = [beta, log(sv), log(su)]

    Returns:
        C : Hessian matrix of the log prior
    """

    theta = np.asarray(theta, dtype=float)

    k = len(theta)

    C = np.zeros((k, k))

    # beta ~ N(0, 100 I)
    C[:k-2, :k-2] = -0.01 * np.eye(k - 2)

    # sv^2 ~ IG(0.00005, 0.00005), theta_v = log(sv)
    C[k-2, k-2] = -4.0 * 0.00005 * np.exp(-2.0 * theta[-2])

    # su ~ IG(1, 0.2877), theta_u = log(su)
    C[k-1, k-1] = -0.2877 * np.exp(-theta[-1])

    return C

##-- normal-half-normal

def lg_pr_nhn_b(theta):
    """
    Log prior and analytic gradient for the Normal-Half-Normal SFA model.

    Parametrization B:
        theta = [beta, log(sigma2), eta]

    where:
        gamma = logistic(eta)
        sv2 = (1 - gamma) * sigma2
        su2 = gamma * sigma2

    Priors:
        beta ~ N(0, 100 I)
        sv2  ~ IG(0.00005, 0.00005)
        su2  ~ IG(5, 10 * log(0.75)^2)

    Includes the log-Jacobian of the transformation:
        (log sigma2, logit gamma) -> (sv2, su2)
    """

    theta = np.asarray(theta, dtype=float).reshape(-1)

    k = len(theta) - 2

    beta = theta[:k]
    log_sigma2 = theta[-2]
    eta = theta[-1]

    # =========================
    # HYPERPARAMETERS
    # =========================
    Vb = 100.0

    a_sv = 0.00005
    b_sv = 0.00005

    a_su = 5.0
    b_su = 10.0 * (np.log(0.75) ** 2)

    # =========================
    # TRANSFORMED PARAMETERS
    # =========================
    sigma2 = np.exp(log_sigma2)

    # Stable logistic
    if eta >= 0:
        gamma = 1.0 / (1.0 + np.exp(-eta))
    else:
        eeta = np.exp(eta)
        gamma = eeta / (1.0 + eeta)

    sv2 = (1.0 - gamma) * sigma2
    su2 = gamma * sigma2

    # =========================
    # LOG PRIOR FOR beta
    # beta ~ N(0, 100 I)
    # =========================
    lgp_b = ( -0.5 * k * np.log(2.0 * np.pi) -0.5 * k * np.log(Vb) - np.dot(beta, beta) / (2.0 * Vb) )

    # =========================
    # LOG PRIOR FOR sv2 ~ IG(a_sv, b_sv)
    # =========================
    lgp_sv2 = ( a_sv * np.log(b_sv) - gammaln(a_sv) - (a_sv + 1.0) * np.log(sv2) - b_sv / sv2 )

    # =========================
    # LOG PRIOR FOR su2 ~ IG(a_su, b_su)
    # =========================
    lgp_su2 = ( a_su * np.log(b_su) - gammaln(a_su) - (a_su + 1.0) * np.log(su2) - b_su / su2 )

    # =========================
    # LOG JACOBIAN
    # |J| = sigma2^2 * gamma * (1 - gamma)
    # =========================
    lgJ = ( 2.0 * log_sigma2 + np.log(gamma) + np.log(1.0 - gamma) )

    # =========================
    # TOTAL LOG PRIOR
    # =========================
    lgp = lgp_b + lgp_sv2 + lgp_su2 + lgJ

    # =========================
    # GRADIENT
    # =========================
    g = np.zeros_like(theta)

    # wrt beta
    g[:k] = -beta / Vb

    # wrt log_sigma2
    d_lgp_sv2_dlogs2 = -(a_sv + 1.0) + b_sv / sv2
    d_lgp_su2_dlogs2 = -(a_su + 1.0) + b_su / su2
    d_lgJ_dlogs2 = 2.0

    g[-2] = d_lgp_sv2_dlogs2 + d_lgp_su2_dlogs2 + d_lgJ_dlogs2

    # wrt eta
    d_lgp_sv2_deta = gamma * (a_sv + 1.0) - gamma * b_sv / sv2
    d_lgp_su2_deta = ( -(1.0 - gamma) * (a_su + 1.0) + (1.0 - gamma) * b_su / su2 )
    d_lgJ_deta = 1.0 - 2.0 * gamma

    g[-1] = d_lgp_sv2_deta + d_lgp_su2_deta + d_lgJ_deta

    return lgp, g

def lg_pr_nhn_a(theta):
    """
    Log prior for the Normal-Half-Normal SFA model.

    Parametrization A:
        theta = [beta, log(sv), log(su)]

    Priors:
        beta ~ N(0, 100 I)
        sv2  ~ IG(0.00005, 0.00005)
        su2  ~ IG(5, 10 * log(0.75)^2)

    where:
        sv = exp(theta[-2])
        su = exp(theta[-1])
        sv2 = sv^2
        su2 = su^2

    The prior is written directly in terms of log(sv) and log(su),
    including the Jacobian terms.
    """

    theta = np.asarray(theta, dtype=float).reshape(-1)

    k = len(theta) - 2

    # Parameters
    beta = theta[:k]
    lv = theta[-2]   # log(sv)
    lu = theta[-1]   # log(su)

    # Hyperparameters
    av = 0.00005
    bv = 0.00005

    au = 5.0
    bu = 10.0 * (np.log(0.75) ** 2)

    # beta prior: beta ~ N(0, 100 I)
    lgp_b = ( -0.5 * k * np.log(2.0 * np.pi * 100.0) -0.5 * np.dot(beta, beta) / 100.0 )

    # sv2 prior: sv2 ~ IG(av, bv), with sv2 = exp(2 * lv)
    # Includes Jacobian log(2 * sv2)
    lgp_sv = ( av * np.log(bv) - gammaln(av) - 2.0 * av * lv - bv * np.exp(-2.0 * lv) + np.log(2.0) )

    # su2 prior: su2 ~ IG(au, bu), with su2 = exp(2 * lu)
    # Includes Jacobian log(2 * su2)
    lgp_su = ( au * np.log(bu) - gammaln(au) - 2.0 * au * lu - bu * np.exp(-2.0 * lu) + np.log(2.0) )

    # Total log prior
    lgp = lgp_b + lgp_sv + lgp_su

    return lgp

def prior_hes_nhn_a(theta):
    """
    Prior Hessian / prior precision matrix for the Normal-Half-Normal SFA model.

    Parametrization A:
        theta = [beta, log(sv), log(su)]

    Returns
    -------
    H_prior : ndarray
        Matrix corresponding to the second derivative contribution
        of the log prior, following the MATLAB implementation.
    """

    theta = np.asarray(theta, dtype=float).reshape(-1)

    k = len(theta) - 2

    # Hyperparameters
    b_sv = 0.00005
    b_su = 10.0 * (np.log(0.75) ** 2)

    H_prior = np.zeros((k + 2, k + 2))

    # beta ~ N(0, 100 I)
    H_prior[:k, :k] = -np.eye(k) / 100.0

    # log(sv), induced by sv^2 ~ IG(0.00005, 0.00005)
    H_prior[k, k] = -4.0 * b_sv * np.exp(-2.0 * theta[k])

    # log(su), induced by su^2 ~ IG(5, 10 * log(0.75)^2)
    H_prior[k + 1, k + 1] = -4.0 * b_su * np.exp(-2.0 * theta[k + 1])

    return H_prior

##-- panel random effects

def lg_pr_re_b(theta):
    """
    Log-prior for the Gaussian panel random-effects model.

    Parametrization B:
        theta = [beta, log(sigma2), logit(gamma)]

    where:
        s2v = (1 - gamma) * sigma2
        s2u = gamma * sigma2

    Returns
    -------
    lgp : float
        Log-prior including the Jacobian term.

    glgp : ndarray
        Gradient of the log-prior.
    """
    k = len(theta) - 2

    beta = theta[:k]
    log_sigma2 = theta[-2]
    eta = theta[-1]

    sigma2 = np.exp(log_sigma2)

    gamma = np.clip( expit(eta), 1e-12, 1.0 - 1e-12, )

    s2v = (1.0 - gamma) * sigma2
    s2u = gamma * sigma2

    # Prior hyperparameters
    av = 0.00005
    bv = 0.00005

    au = 0.5
    bu = 0.00005

    # beta ~ N(0, 100I)
    lgp_beta = ( -0.5 * k * np.log(2.0 * np.pi) -0.5 * k * np.log(100.0) -0.5 * (beta @ beta) / 100.0 )

    g_beta = -beta / 100.0

    # Inverse-gamma priors
    lgp_s2v = ( av * np.log(bv) - gammaln(av) - (av + 1.0) * np.log(s2v) - bv / s2v )

    lgp_s2u = ( au * np.log(bu) - gammaln(au) - (au + 1.0) * np.log(s2u) - bu / s2u )

    # Jacobian:
    # |d(s2v, s2u) / d(log_sigma2, logit_gamma)| = s2v * s2u
    log_jac = np.log(s2v) + np.log(s2u)

    lgp = float( lgp_beta + lgp_s2v + lgp_s2u + log_jac )

    # Derivatives with respect to log(s2v) and log(s2u)
    d_lgp_s2v_dlog_s2v = -(av + 1.0) + bv / s2v
    d_lgp_s2u_dlog_s2u = -(au + 1.0) + bu / s2u

    # Jacobian contributes +1 to each derivative
    qv = d_lgp_s2v_dlog_s2v + 1.0
    qu = d_lgp_s2u_dlog_s2u + 1.0

    # Chain rule to parametrization B
    g_log_sigma2 = qv + qu

    g_eta = ( -gamma * qv + (1.0 - gamma) * qu )

    glgp = np.concatenate( ( g_beta, np.array([ g_log_sigma2, g_eta, ]), ) )

    return lgp, glgp


def lg_pr_re_a(theta):
    """
    Log-prior for the Gaussian panel random-effects model.

    Parametrization A:
        theta = [beta, log(s_v), log(s_alpha)]

    Priors:
        beta      ~ N(0, 100 I)
        s_v^2     ~ IG(av, bv)
        s_alpha^2 ~ IG(aa, ba)
    """
    k = len(theta) - 2

    beta = theta[:k]
    log_sv = theta[-2]
    log_sa = theta[-1]

    av = 0.00005
    bv = 0.00005

    aa = 0.5
    ba = 0.00005

    # beta ~ N(0, 100I)
    lgp_beta = ( -0.5 * k * np.log(2.0 * np.pi * 100.0) -0.5 * (beta @ beta) / 100.0 )

    # s_v^2 ~ IG(av, bv), transformed to log(s_v)
    lgp_sv = ( av * np.log(bv) - gammaln(av) - (av + 1.0) * (2.0 * log_sv) - bv * np.exp(-2.0 * log_sv) + np.log(2.0) + 2.0 * log_sv )

    # s_alpha^2 ~ IG(aa, ba), transformed to log(s_alpha)
    lgp_sa = ( aa * np.log(ba) - gammaln(aa) - (aa + 1.0) * (2.0 * log_sa) - ba * np.exp(-2.0 * log_sa) + np.log(2.0) + 2.0 * log_sa )

    return float(lgp_beta + lgp_sv + lgp_sa)


def prior_hes_re_a(theta):
    """
    Hessian of the log-prior for the Gaussian panel
    random-effects model.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]
    """
    k = len(theta) - 2

    bv = 0.00005
    bu = 0.00005

    s2v = np.exp(2.0 * theta[-2])
    s2u = np.exp(2.0 * theta[-1])

    Hlgp = np.zeros((k + 2, k + 2))

    # beta ~ N(0, 100I)
    Hlgp[:k, :k] = -(1.0 / 100.0) * np.eye(k)

    # Prior contribution for log(s_v)
    Hlgp[k, k] = -4.0 * bv / s2v

    # Prior contribution for log(s_u)
    Hlgp[k + 1, k + 1] = -4.0 * bu / s2u

    return Hlgp
