# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:23:19 2026

@author: Kamil Makieła, Krakow University of Economics
"""

# standard packages 
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve #, solve_triangular
from types import SimpleNamespace
from scipy.special import expit

#internal modules
from .logLikelihoods import lgl_kmnrl, nlgl_nex_b, nlgl_nhn_b, nlgl_nexP_b, nlgl_nhnP_b, nlgl_re_b
from .posteriors import nlgMAP_kmnrl, lgMAP_kmnrl, nlgMAP_nex_b, lgMAP_nex_a, nlgMAP_nhn_b, lgMAP_nhn_a
from .posteriors import nlgMAP_nexP_b, lgMAP_nexP_a, nlgMAP_nhnP_b, lgMAP_nhnP_a, nlgMAP_re_b, lgMAP_re_a
from .priors import prior_hes_kmnrl, prior_hes_nex_a, prior_hes_nhn_a, prior_hes_re_a
from .scores import scores_kmnrl, scores_nex_a, scores_nhn_a, scores_nexP_a, scores_nhnP_a, scores_re_a

class SFAResults(SimpleNamespace):
    """Container for the results returned by :func:`fit`."""

    def summary(self, level=0.95, decimals=4):
        """
        Print and return a summary of one fitted model.

        Maximum-likelihood results are always reported. If Bayesian
        results are available (fit called with if_mdd=1), a separate
        section reports the log posterior at the mode, log marginal
        data density, posterior-mode estimates, posterior standard
        deviations, and approximate credible intervals.
        """
        import pandas as pd
        from scipy.stats import norm

        if not 0.0 < level < 1.0:
            raise ValueError('level must lie strictly between 0 and 1.')

        alpha = 1.0 - level
        critical_value = norm.ppf(1.0 - alpha / 2.0)
        tail = 100.0 * alpha / 2.0

        names = getattr(self, 'param_names', [f"param_{i}" for i in range(len(self.params_ml))],)

        n_scale = 1 if self.name == 'cnlrm' else 2
        scale_idx = np.arange(len(self.params_ml) - n_scale, len(self.params_ml))

        def make_table(estimates, dispersion, theta, theta_dispersion, dispersion_label,):
            estimates = np.asarray(estimates)
            dispersion = np.asarray(dispersion)
            theta = np.asarray(theta)
            theta_dispersion = np.asarray(theta_dispersion)

            lower = estimates - critical_value * dispersion
            upper = estimates + critical_value * dispersion

            # Scale parameters are stored as exp(log scale). Construct
            # intervals on the log scale and exponentiate the endpoints.
            lower[scale_idx] = np.exp(theta[scale_idx] - critical_value * theta_dispersion[scale_idx])
            upper[scale_idx] = np.exp(theta[scale_idx] + critical_value * theta_dispersion[scale_idx])

            return pd.DataFrame({'Estimate': estimates, dispersion_label: dispersion, f'{tail:.1f}%': lower, f'{100.0 - tail:.1f}%': upper,}, index=names,)

        ml_table = make_table(self.params_ml, self.params_ml_se, self.theta_ml, self.theta_ml_se, 'Std. Error',)

        model_labels = {
            'cnlrm': 'Classical normal linear regression',
            'nex': 'Normal-exponential stochastic frontier',
            'nhn': 'Normal-half-normal stochastic frontier',
            'nexp': 'Panel normal-exponential stochastic frontier',
            'nhnp': 'Panel normal-half-normal stochastic frontier',
            'RE': 'Gaussian panel random-effects model',
        }

        title = model_labels.get(self.name, self.name)
        width = 68

        print('=' * width)
        print('Stochastic Frontier Model Results')
        print('=' * width)
        print(f'Model:              {title}')
        print(f'Observations:       {self.nobs}')
        print(f'Units:              {self.n}')
        print(f'Time periods:       {self.T}')
        print(f'Parameters:         {self.k}')
        print(f'Log-likelihood:     {self.l_max:.{decimals}f}')
        print(f'AIC:                {self.aic:.{decimals}f}')
        print(f'BIC:                {self.bic:.{decimals}f}')

        print('-' * width)
        print(f'ML estimates with {100.0 * level:.1f}% confidence intervals (Wald-type)')
        print('-' * width)
        print(ml_table.round(decimals).to_string())

        output = {'ml': ml_table}

        if hasattr(self, 'bayes'):
            bayes_table = make_table(self.bayes.param_post, self.bayes.param_post_se, self.bayes.theta_post, self.bayes.theta_post_se, 'Post. SD',)

            print('=' * width)
            print('Bayesian Results')
            print('=' * width)
            print(f'Log posterior mode:   {self.p_max:.{decimals}f}')
            print(f'Log marginal density: {self.mdd:.{decimals}f}')
            print('-' * width)
            print(f'Posterior estimates with {100.0 * level:.1f}% credible intervals (Laplace appr.)')
            print('-' * width)
            print(bayes_table.round(decimals).to_string())

            output['bayes'] = bayes_table

        print('=' * width)
        print('NOTE: to get (in)efficiency scores run "inefficiency" module')

        return output


def fit(X, y, n, T=1, sfa_opt=1, dec_crit=0, if_mdd=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    nT = n * T

    # MATLAB: b_ml = X \ y
    b_ml = np.linalg.lstsq(X, y, rcond=None)[0]

    e_ml = y - X @ b_ml
    s_ml = np.sqrt(e_ml @ e_ml / nT)

    k = X.shape[1] + 2
    sig_idx = np.array([k - 2, k - 1])   # Python equivalent of MATLAB k-1:k
    
    model = SFAResults()
    model.sfa_opt = sfa_opt
    model.nobs = nT
    model.n = n
    model.T = T

    if sfa_opt == 0:
        model.name = 'cnlrm'
        k = k - 1
        sig_idx = np.array([k - 1])

        theta_l_max = np.r_[b_ml, np.log(s_ml)]
        theta_start = theta_l_max.copy()
        theta_l_max_a = theta_l_max.copy()

        l_max, _ = lgl_kmnrl(theta_l_max, X, y)

        Var_a = np.zeros((k, k))
        Var_a[:-1, :-1] = s_ml**2 * inv_spd_chol_upper(X.T @ X)
        Var_a[-1, -1] = 1 / (2 * nT)

        funopt_post = lambda th: nlgMAP_kmnrl(th, X, y)
        fun_MAP_a = lambda th: lgMAP_kmnrl(th, X, y)
        scores = lambda th: scores_kmnrl(th, X, y)
        prior_hess = lambda th: prior_hes_kmnrl(th)

        model.converged = True
        model.optimizer_message = 'Closed-form maximum-likelihood solution.'
        model.iterations = 0
        model.function_evaluations = 1

    elif sfa_opt == 1:
        model.name = 'nex'
        theta_start = np.r_[b_ml, 2 * np.log(s_ml), 0.0]
        funopt = lambda th: nlgl_nex_b(th, X, y)
        funopt_post = lambda th: nlgMAP_nex_b(th, X, y)
        fun_MAP_a = lambda th: lgMAP_nex_a(th, X, y)
        scores = lambda th: scores_nex_a(th, X, y)
        prior_hess = lambda th: prior_hes_nex_a(th)

    elif sfa_opt == 2:
        model.name = 'nhn'
        theta_start = np.r_[b_ml, 2 * np.log(s_ml), 0.0]
        funopt = lambda th: nlgl_nhn_b(th, X, y)
        funopt_post = lambda th: nlgMAP_nhn_b(th, X, y)
        fun_MAP_a = lambda th: lgMAP_nhn_a(th, X, y)
        scores = lambda th: scores_nhn_a(th, X, y)
        prior_hess = lambda th: prior_hes_nhn_a(th)

    elif sfa_opt == 3:
        model.name = 'nexp'
        theta_start = np.r_[b_ml, np.log(s_ml), 0.0]
        funopt = lambda th: nlgl_nexP_b(th, X, y, n, T)
        funopt_post = lambda th: nlgMAP_nexP_b(th, X, y, n, T)
        fun_MAP_a = lambda th: lgMAP_nexP_a(th, X, y, n, T)
        scores = lambda th: scores_nexP_a(th, X, y, n, T)
        prior_hess = lambda th: prior_hes_nex_a(th)

    elif sfa_opt == 4:
        model.name = 'nhnp'
        theta_start = np.r_[b_ml, np.log(s_ml), 0.0]
        funopt = lambda th: nlgl_nhnP_b(th, X, y, n, T)
        funopt_post = lambda th: nlgMAP_nhnP_b(th, X, y, n, T)
        fun_MAP_a = lambda th: lgMAP_nhnP_a(th, X, y, n, T)
        scores = lambda th: scores_nhnP_a(th, X, y, n, T)
        prior_hess = lambda th: prior_hes_nhn_a(th)

    elif sfa_opt == 5:
        model.name = 'RE'
        theta_start = np.r_[b_ml, 2 * np.log(s_ml), 0.0]
        funopt = lambda th: nlgl_re_b(th, X, y, n, T)
        funopt_post = lambda th: nlgMAP_re_b(th, X, y, n, T)
        fun_MAP_a = lambda th: lgMAP_re_a(th, X, y, n, T)
        scores = lambda th: scores_re_a(th, X, y, n, T)
        prior_hess = lambda th: prior_hes_re_a(th)

    else:
        raise ValueError('Unknown sfa_opt.')

    if sfa_opt != 0:
        res = minimize(funopt, theta_start, method='BFGS', jac=True, options={'maxiter': 5000, 'gtol': 1e-12, 'disp': False,},)

        theta_l_max = res.x
        l_max = -res.fun

        model.converged = bool(res.success)
        model.optimizer_message = str(res.message)
        model.iterations = getattr(res, 'nit', None)
        model.function_evaluations = getattr(res, 'nfev', None)

        theta_l_max_a = theta_l_max.copy()
        theta_l_max_a[-2], theta_l_max_a[-1] = sv_su_from_sigma_gamma(theta_l_max_a[-2], theta_l_max_a[-1],)

        G_lik = scores(theta_l_max_a)
        hes = G_lik.T @ G_lik
        hes = stabilize_hessian(hes)
        Var_a = np.linalg.inv(hes)


    model.bic = k * np.log(nT) - 2 * l_max
    model.aic = 2 * k - 2 * l_max
    model.l_max = l_max

    if if_mdd == 1:
        theta_l_max_hlp = theta_l_max.copy()
        theta_l_max_hlp[-1] = np.clip(theta_l_max[-1], -2.197224577, 2.197224577)

        try:
            res_post = minimize(funopt_post, theta_l_max_hlp, method='BFGS', jac=True, options={'maxiter': 5000, 'gtol': 1e-12, 'disp': False},)
        except Exception:
            res_post = minimize(funopt_post, theta_start, method='BFGS', jac=True, options={'maxiter': 5000, 'gtol': 1e-12, 'disp': False},)

        theta_p_max = res_post.x

        theta_p_a = theta_p_max.copy()
        if sfa_opt != 0:
            theta_p_a[-2], theta_p_a[-1] = sv_su_from_sigma_gamma(theta_p_max[-2], theta_p_max[-1],)

        p_max_a = fun_MAP_a(theta_p_a)

        G_a = scores(theta_p_a)
        H_pr_a = prior_hess(theta_p_a)

        hes_a = G_a.T @ G_a - H_pr_a
        hes_a = stabilize_hessian(hes_a)

        R = cholesky(hes_a, lower=False)
        logdetHa = 2 * np.sum(np.log(np.diag(R)))

        model.mdd = p_max_a + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdetHa
        model.p_max = p_max_a

    else:
        model.p_max = l_max
        model.mdd = -0.5 * model.bic

    if dec_crit == 0:
        model.inf_cr = model.bic
        model.mdd = -0.5 * model.bic
    elif dec_crit == 1:
        model.inf_cr = -model.mdd
    elif dec_crit == 2:
        model.inf_cr = model.aic
        model.mdd = -0.5 * model.bic
    else:
        raise ValueError('Unknown dec_crit.')

    model.k = k

    n_beta = X.shape[1]
    beta_names = [f'beta_{j}' for j in range(n_beta)]

    if model.name == 'cnlrm':
        scale_names = ['sigma_v']
    elif model.name == 'RE':
        scale_names = ['sigma_v', 'sigma_alpha']
    else:
        scale_names = ['sigma_v', 'sigma_u']

    model.param_names = beta_names + scale_names

    model.theta_Var_ml = Var_a
    model.theta_ml = theta_l_max_a
    model.theta_ml_se = np.sqrt(np.diag(Var_a))
    model.theta_b = theta_l_max

    model.params_ml = theta_l_max_a.copy()
    model.params_ml_se = model.theta_ml_se.copy()

    model.params_ml[sig_idx] = np.exp(theta_l_max_a[sig_idx])
    model.params_ml_se[sig_idx] = (model.params_ml[sig_idx] * model.theta_ml_se[sig_idx])
    if if_mdd == 1:
        model.bayes = SimpleNamespace()

        model.bayes.theta_Var_post = inv_spd_chol_upper(hes_a)

        model.bayes.theta_post = theta_p_a
        model.bayes.theta_post_se = np.sqrt(np.diag(model.bayes.theta_Var_post))

        model.bayes.param_post = theta_p_a.copy()
        model.bayes.param_post_se = model.bayes.theta_post_se.copy()

        model.bayes.param_post[sig_idx] = np.exp(theta_p_a[sig_idx])
        model.bayes.param_post_se[sig_idx] = (model.bayes.param_post[sig_idx] * model.bayes.theta_post_se[sig_idx])

    return model

def stabilize_hessian(H, eps=1e-10):
    H = 0.5 * (H + H.T)
    d, V = np.linalg.eigh(H)
    d = np.maximum(d, eps)
    return V @ np.diag(d) @ V.T


def inv_spd_chol_upper(A):
    """
    Inverse of a symmetric positive definite matrix A using upper Cholesky.
    Equivalent to MATLAB: R = chol(A); invA = R \\ (R' \\ eye(k));
    """
    R = cholesky(A, lower=False)
    I = np.eye(A.shape[0])
    return cho_solve((R, False), I)

def sv_su_from_sigma_gamma(log_sigma2, eta):
    """
    Convert from (log sigma^2, eta) to (log s_v, log s_u),
    where gamma = logistic(eta).
    """

    gamma = expit(eta)
    gamma = np.clip(gamma, 1e-12, 1.0 - 1e-12)

    log_sv = 0.5 * (log_sigma2 + np.log1p(-gamma))
    log_su = 0.5 * (log_sigma2 + np.log(gamma))

    return log_sv, log_su