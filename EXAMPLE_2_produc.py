# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:23:19 2026
@author: makielak
"""

import pandas as pd
import numpy as np 

import sfa2
import inefficiency

# The dataset is from PLM package available in R (dataset: Produc), references: 
# Croissant, Y., & Millo, G. (2008). Panel Data Econometrics in R: The plm Package. Journal of Statistical Software, 27(2), 1–43

# In the excel file, we can choose one of the three datasets:
# OD - the original dataset (the explanatory variables are not centered)
# m0 - this is the basic dataset from PLM package with logged variables 
#   the explanatory variables are centered
# m1 - this is for translog forntier specification
# m_opt - this is the optimal frontier specification based on SF-BMA package available in MATLAB


#sf_op = 0;   # Gaussian model (classic regression)
sf_op = 1;     # normal-expnential
#sf_op = 2;    # normal-half-normal
#sf_op = 3;    # normal-exponential
#sf_op = 4;    # panel normal-expnential 
#sf_op = 5;    # panel RE

data = pd.read_excel('Produc.xlsx', header=0, sheet_name='m1')
y = np.asarray(data.iloc[:,3], dtype=float).reshape(-1)
X = np.asarray(data.iloc[:,4:], dtype=float)

# to run the fit() method below you need at least:
#   X - independent variables (including the constant term)
#   y - the dependent
#   n - the number of units in the datasets (datapoints in cross-section data)
# NOTE: traditionally in SFA, variables in X and y are in natural logs. 

# additionally fit() method may take:
#   T - the number of time periods (must be specified for panel data models).
#   sfa_opt - type of sf model in numbers; the default is 1, which is 
#     normal-exponential. 
#   dec_crit - decision criteria (for future use); the default is 0, 
#     which is BIC; dec_crit=1 is mdd (marignal likelihood). 
#   if_mdd - whether or not to estimate the Bayesian model, and thus the 
#     marginal likelihood (mdd); the default is 0; in such case (0) mdd is 
#     simply mdd~-0.5BIC, so a very rough approximation and it is not
#     provided in the summary() method.
# NOTE: I suggest using dec_crit=1 with if_mdd=1, otherwise mdd is 
#   approximated using BIC, which is likely not that accurate.
# NOTE: if you use non-panel (pooled) models on panel data I suggest 
#   setting n as the total number of datapoints and T=1
my_n=48
my_T=17

print('Example based on Produc dataset available in R (plm package)')
model  = sfa2.fit(X, y, my_n, T=my_T, sfa_opt=sf_op, dec_crit=1, if_mdd=1)
model.summary();

# this part is fairly automated, just provide the fitted model and the data
u, te = inefficiency.jondrow(model, X, y)



