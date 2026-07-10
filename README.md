# Python-SFA
Stochastic Frontier Analysis with Python

Included SFA models:
- simple Gausssian (non-sfa: classic normal linear regression, cnlrm)
- normal-exponential (nex)
- normal-half-normal (nhn)
- panel normal-exponential (nex)
- panel normal-half-normal (nhn)
- panel Gaussian model (non-sfa: random effects, RE)

The SFA codes provided here are translations from a larger MATLAB SF-BMA project. More will come in due time. The codes allow to estimate one of the abovementioned SF models using maximum likelihood and Bayesian inference (i.e., when: if_mdd=1). There are two main modules: sfa2 and inefficiency (module 'sfa' is also there but it does not have the summary() method; I leave it for future purposes). I keep the two estimations separate (model parameters and (in)efficiencies) for a reason: if the number of inefficiency socres is large it slows down runtime significantly. I think it is important to build your frontier first and then estimate efficiencies. 

See comments in 'EXAMPLE_script' for details. 
