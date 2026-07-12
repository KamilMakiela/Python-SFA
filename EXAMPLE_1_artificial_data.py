# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:23:19 2026
@author: makielak
"""

import pandas as pd
import numpy as np 

import sfa2
import inefficiency

# We can choose one of the artificial datasets available in Excel for each 
# model. Description of the data generating process (DGP) is in the Excel file
# Just uncomment the desired option below

#sf_op = 0; sf_label = 'cnlrm'; my_n = 500; my_T = 1
sf_op = 1; sf_label = 'nex'; my_n = 500; my_T = 1
#sf_op = 2; sf_label = 'nhn'; my_n = 500; my_T = 1
#sf_op = 3; sf_label = 'nexp'; my_n = 50; my_T = 10
#sf_op = 4; sf_label = 'nhnp'; my_n = 50; my_T = 10
#sf_op = 5; sf_label = 'RE'; my_n = 50; my_T = 10

data = pd.read_excel('dane.xlsx', header=None, sheet_name=sf_label)
y = np.asarray(data.iloc[:,0], dtype=float).reshape(-1)
X = np.asarray(data.iloc[:,1:], dtype=float)

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

print('Example based on artificial data')
model  = sfa2.fit(X, y, my_n, T=my_T, sfa_opt=sf_op, dec_crit=1, if_mdd=1)
model.summary();

# this part is fairly automated, just provide the fitted model and the data
u1, te1 = inefficiency.jondrow(model, X, y)



