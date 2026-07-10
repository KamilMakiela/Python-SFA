# Python-SFA
Stochastic Frontier Analysis with Python

Included SFA models:
- simple Gausssian (classic normal linear regression, cnlrm)
- normal-exponential (nex)
- normal-half-normal (nhn)
- panel normal-exponential (nex)
- panel normal-half-normal (nhn)
- panel Gaussian model (random effects, RE)

The SFA codes provided here are translations from a larger MATLAB SF-BMA project. More will come in due time. The codes allow to estimate one of the abovementioned SF models using maximum likelihood and Bayesian inference (i.e., when: if_mdd=1). There are two modules: sfa2 and ineefficiency (module 'sfa' is also there but it does not have the summary() method; I keep it for future purposes). 
See comments in 'EXAMPLE_script' for details. 
