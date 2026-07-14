# Python-SFA

Stochastic Frontier Analysis in Python.

This package provides tools for estimating cross-sectional and panel-data stochastic frontier models using maximum likelihood and Bayesian inference.

## Available Models

The following models are currently implemented:

* **Classical normal linear regression (`cnlrm`)**
  Standard Gaussian linear regression without an inefficiency component.

* **Normal-exponential stochastic frontier (`nex`)**
  Cross-sectional stochastic frontier model with normally distributed statistical noise and exponentially distributed inefficiency.

* **Normal-half-normal stochastic frontier (`nhn`)**
  Cross-sectional stochastic frontier model with normally distributed statistical noise and half-normally distributed inefficiency.

* **Panel normal-exponential stochastic frontier**
  Panel-data stochastic frontier model with exponentially distributed unit-specific inefficiency.

* **Panel normal-half-normal stochastic frontier**
  Panel-data stochastic frontier model with half-normally distributed unit-specific inefficiency.

* **Panel Gaussian random-effects model (`RE`)**
  Standard Gaussian random-effects model without a one-sided inefficiency component.

## Project Background

The Python implementation is based on a code developed for a larger MATLAB project on stochastic frontier Bayesian model averaging. See Makieła (2026) for details (reference below). The current package focuses primarily on a single-model estimation. Additional model-search and Bayesian model-averaging procedures will added in future versions.

## Main Modules

The package currently contains two main modules:

* `sfa2` — estimation of model parameters and presentation of estimation results;
* `inefficiency` — estimation of inefficiency and technical efficiency scores.

An additional module named `sfa` is also included. It contains an earlier implementation without the `summary()` method. It is retained for future development of model search algorithms. Parameter estimation and efficiency estimation are separated on purpose. Once the number of observations is large, computing inefficiency scores can increase runtime substantially. This design allows users to:

1. estimate and evaluate the stochastic frontier model;
2. select the preferred model specification;
3. compute inefficiency and efficiency scores only for the selected model.

In mmy experience, it is better to construct and assess the production frontier before estimating the efficiency scores.

## Estimating a Model

The main estimation function is the module-level function `fit()` from the `sfa2` module:

```python
from sfa2 import fit

model = fit(X, y, n, T, sfa_opt, dec_crit, if_mdd)
```

The function estimates a single stochastic frontier model (or a simple regression) and returns a model-results object containing parameter estimates, standard errors, model-fit measures, etc. Depending on the selected options, the returned object may include:

* maximum-likelihood estimates;
* Bayesian posterior estimates;
* Akaike information criterion;
* Bayesian information criterion;
* maximized log-likelihood;
* maximized lop-posterior; 
* integrated likelihood, also known as the marginal data density or marginal likelihood;
* additional model diagnostics and summary statistics.

The integrated likelihood provides a fully Bayesian measure of model fit and can later be used by model-search or Bayesian model-averaging algorithms (prospective future use).

## The `summary()` Method

Models estimated with `sfa2.fit()` provide a `summary()` method:

```python
model.summary()
```

The method prints a formatted summary of the estimation results, including parameter estimates, standard errors, confidence or credible intervals, and model-fit statistics. When the marginal data density is available, the summary also includes a separate Bayesian section with posterior estimates and the relevant Bayesian model-fit measures. Because `summary()` prints the results directly, it can be called without wrapping it in `print()`:

```python
model.summary()
```

rather than:

```python
print(model.summary())
```

## Function Arguments

### `X`

The regressor matrix.

`X` must already contain a column of ones when an intercept is required. It is recommended that the intercept be placed in the first column:

```python
X = np.column_stack((np.ones(len(y)), regressors))
```

Keeping the constant (intercept column) in the first column from the left is especially important when `X` is later supplied to model-search procedures that distinguish between mandatory and optional regressors.

### `y`

The dependent variable.

It should contain observations in the same order as the rows of `X`.

### `n`

The number of cross-sectional units.

For cross-sectional models, this will typically equal the total number of observations. For panel models, `n` is the number of units.

### `T`

The number of time periods.

For cross-sectional models, use:

```python
T = 1
```

For panel models, the data should be arranged consistently with the panel structure expected by the estimation functions: unit1->time, unit2->time, and so on (see example 2). 

### `sfa_opt`

Selects the model specification:

| Value | Model                                                  |
| ----: | ------------------------------------------------------ |
|   `0` | Classical normal linear regression (`cnlrm`)           |
|   `1` | Cross-sectional normal-exponential stochastic frontier |
|   `2` | Cross-sectional normal-half-normal stochastic frontier |
|   `3` | Panel normal-exponential stochastic frontier           |
|   `4` | Panel normal-half-normal stochastic frontier           |
|   `5` | Panel Gaussian random-effects model                    |

### `dec_crit`

Selects the main model-fit criterion.

Recommended settings are:

```python
dec_crit = 1
```

for the integrated likelihood, or:

```python
dec_crit = 0
```

for the Bayesian information criterion.

The integrated likelihood provides a fully Bayesian basis for model comparison, whereas BIC is considerably faster to compute.

### `if_mdd`

Controls whether Bayesian calculations and the marginal data density are computed:

```python
if_mdd = 1
```

enables Bayesian inference and integrated likelihood calculations.

```python
if_mdd = 0
```

uses maximum-likelihood estimation without calculating the marginal data density.

For a fully Bayesian analysis, I recommend:

```python
dec_crit = 1
if_mdd = 1
```

For faster estimation based on BIC, use:

```python
dec_crit = 0
if_mdd = 0
```

## Example

```python
import numpy as np

from sfa2 import fit

# Dependent variable
y = np.asarray(y, dtype=float).reshape(-1)

# Regressor matrix with the intercept in the first column
X = np.column_stack((np.ones(len(y)), x1, x2, x3))

# Cross-sectional normal-exponential stochastic frontier
n = len(y)
T = 1
sfa_opt = 1

# Fully Bayesian model evaluation
dec_crit = 1
if_mdd = 1

model = fit(X, y, n, T, sfa_opt, dec_crit, if_mdd)
model.summary()

```

## Efficiency Estimation

Efficiency and inefficiency scores are computed separately using the functions provided in the `inefficiency` module. For now, the algorithm is an implementation of a well-known Jondrow et al. (1982) estimator. 

This separation avoids unnecessary runtime when competing frontier specifications need to be first estimated and compared.

A typical workflow:

1. estimate several candidate stochastic frontier models;
2. compare them using BIC or integrated likelihood;
3. retain the preferred specification;
4. calculate inefficiency and technical efficiency scores for that model.

## Example

```Python
# once me have 'model' we can estimate (in)effciency scores as
# bayes = 0  # scores based on ML
beyes = 1    # scores based on Bayesian estimate

u, ef = inefficiency.jondrow(model, X, y, bayes)
```

## Other examples

For complete examples with simulated and real data, including data preparation, model estimation, result summary(), efficiency-score calculation and more comments, see:

```text
EXAMPLE_1_artificial_data
EXAMPLE_2_produc
```

## Development Status

This package is under development. The current implementation is based on a broader MATLAB project for stochastic frontier and Bayesian model averaging. Additional features and model-search procedures will be added in future releases.

## References

Jondrow, J., Knox Lovell, C.A., Materov, I.S. and Schmidt, P. (1982), “On the estimation of technical inefficiency in the stochastic frontier production function model”, Journal of Econometrics, Vol. 19 No. 2–3, pp. 233–238, doi: 10.1016/0304-4076(82)90004-5.

Makieła, K. (2026). Model uncertainty under non-Gaussian errors: Bayesian model averaging and selection in stochastic frontier models. Forthcoming.
