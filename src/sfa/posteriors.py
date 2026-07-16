# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 12:11:25 2026

@author: makielak
"""

from .logLikelihoods import lgl_kmnrl
from .priors import lg_pr_kmnrl



def nlgMAP_kmnrl(theta, X, y):
    l1, g1 = lgl_kmnrl(theta, X, y)
    l2, g2 = lg_pr_kmnrl(theta)

    nmap = -l1 - l2
    ng = -g1 - g2

    return nmap, ng


def lgMAP_kmnrl(theta, X, y):
    """
    Log posterior for the classical normal linear regression model.

    theta = [beta, log(sv)]
    """

    a, _ = lgl_kmnrl(theta, X, y)
    b, _ = lg_pr_kmnrl(theta)

    lgmap = a + b

    return lgmap

##-- normal-exponential 

from .logLikelihoods import nlgl_nex_b
from .priors import lg_pr_nex_b


def nlgMAP_nex_b(theta, X, y):
    """
    Negative log posterior for the normal-exponential SFA model.

    Parametrization B:
        theta = [beta, log_sigma2, eta]
    """

    nll, g_nll = nlgl_nex_b(theta, X, y)
    lp, g_lp = lg_pr_nex_b(theta)

    nmap = nll - lp
    ng = g_nll - g_lp

    return nmap, ng


from .logLikelihoods import lgl_nex_a
from .priors import lg_pr_nex_a


def lgMAP_nex_a(theta, X, y):
    """
    Log posterior for the normal-exponential SFA model.

    Parametrization A:
        theta = [beta, log(sv), log(su)]
    """

    lgmap = lgl_nex_a(theta, X, y) + lg_pr_nex_a(theta)

    return lgmap

##-- normal-half-normal

from .logLikelihoods import nlgl_nhn_b
from .priors import lg_pr_nhn_b

def nlgMAP_nhn_b(theta, X, y):
    """
    Negative log-posterior / negative log-MAP objective
    for the Normal-Half-Normal SFA model.

    Parametrization B:
        theta = [beta, log_sigma2, eta]

    Computes:
        nmap = negative log-likelihood - log-prior
    """

    nL, ngrad_lgl = nlgl_nhn_b(theta, X, y)
    lgp, grad_lgp = lg_pr_nhn_b(theta)

    nmap = nL - lgp
    ng = ngrad_lgl - grad_lgp

    return nmap, ng

from .logLikelihoods import lgl_nhn_a
from .priors import lg_pr_nhn_a


def lgMAP_nhn_a(theta, X, y):
    """
    Log-posterior / log-MAP objective for the Normal-Half-Normal SFA model.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]

    Computes:
        lgmap = log-likelihood + log-prior
    """

    lgmap = lgl_nhn_a(theta, X, y) + lg_pr_nhn_a(theta)

    return lgmap

##-- Panel normal-exponential 

from .logLikelihoods import nlgl_nexP_b
#from priors import lg_pr_nex_b # the same as in nex

def nlgMAP_nexP_b(theta, X, y, n, T):
    """
    Negative log-posterior for the panel normal-exponential SFA model.

    Parametrization B:
        theta = [beta, log(sigma^2), logit(gamma)]

    Returns
    -------
    nmap : float
        Negative log-posterior.
    gradient : ndarray
        Gradient of the negative log-posterior.
    """
    nll, grad_nll = nlgl_nexP_b(theta,X,y,n,T)

    log_prior, grad_log_prior = lg_pr_nex_b(theta)

    nmap = nll - log_prior
    gradient = grad_nll - grad_log_prior

    return nmap, gradient

from .logLikelihoods import lgl_nexP_a
#from priors import lg_pr_nex_a #the same as in non-panel n-ex


def lgMAP_nexP_a(theta, X, y, n, T):
    """
    Log-posterior for the panel normal-exponential SFA model.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]
    """
    return ( lgl_nexP_a(theta, X, y, n, T) + lg_pr_nex_a(theta) )

##-- panel normal-half-normal

from .logLikelihoods import nlgl_nhnP_b
#from priors import lg_pr_nhn_b

def nlgMAP_nhnP_b(theta, X, y, n, T):
    """
    Negative log-posterior for panel normal-half-normal SFA.

    Parametrization B:
        theta = [beta, log(sigma2), logit(gamma)]
    """
    nll, g_nll = nlgl_nhnP_b(theta, X, y, n, T)
    lp, g_lp = lg_pr_nhn_b(theta)

    nmap = nll - lp
    ng = g_nll - g_lp

    return nmap, ng

from .logLikelihoods import lgl_nhnP_a
#from priors import lg_pr_nhn_a

def lgMAP_nhnP_a(theta, X, y, n, T):
    """
    Log-posterior for panel normal-half-normal SFA.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]
    """
    return ( lgl_nhnP_a(theta, X, y, n, T) + lg_pr_nhn_a(theta) )


##-- panel RE model

from .logLikelihoods import nlgl_re_b
from .priors import lg_pr_re_b


def nlgMAP_re_b(theta, X, y, n, T):
    """
    Negative log-posterior for the Gaussian panel random-effects model.

    Parametrization B:
        theta = [beta, log(sigma2), logit(gamma)]
    """
    nL, ng = nlgl_re_b(theta, X, y, n, T)
    lgp, glgp = lg_pr_re_b(theta)

    nmap = nL - lgp
    ng = ng - glgp

    return nmap, ng

from .logLikelihoods import lgl_re_a
from .priors import lg_pr_re_a


def lgMAP_re_a(theta, X, y, n, T):
    """
    Log-posterior for the Gaussian panel random-effects model.

    Parametrization A:
        theta = [beta, log(s_v), log(s_u)]
    """
    return ( lgl_re_a(theta, X, y, n, T) + lg_pr_re_a(theta) )
