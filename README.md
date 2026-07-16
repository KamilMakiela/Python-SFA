# Stochastic Frontier Analysis

A Python package for estimating cross-sectional and panel data stochastic frontier models using maximum likelihood and Bayesian methods.

## Features

The package currently supports:

- classical normal linear regression;
- cross-sectional normal-exponential stochastic frontier models;
- cross-sectional normal-half-normal stochastic frontier models;
- panel normal-exponential stochastic frontier models;
- panel normal-half-normal stochastic frontier models;
- panel random-effects models;
- maximum likelihood estimation;
- Bayesian estimation;
- marginal data density calculations (aka integrated likelihood);
- inefficiency and efficiency estimation.

The current release focuses on estimating individual stochastic frontier models. Model search and Bayesian model averaging procedures will be added in future versions.

## Installation

Install the package from PyPI with:

```bash
pip install stochastic-frontier-analysis
```

For local development, clone the repository and install it in editable mode from the project root:

```bash
python -m pip install -e .
```

To install the optional dependencies used by the examples:

```bash
python -m pip install -e ".[examples]"
```

Although the distribution name is `stochastic-frontier-analysis`, the Python import package is NAMED short: `sfa`.

## Package Structure

The main public modules are:

- `sf_model` — model estimation and presentation of estimation results;
- `inefficiency` — estimation of inefficiency and efficiency scores.

They can be imported with:

```python
from sfa import sf_model, inefficiency
```

Model and efficiency estimation is kept separate. This allows users to estimate and compare alternative SF specifications before computing efficiency scores for the preferred model, which may be quite demanding.

## Available Models

The argument `sfa_opt` selects the model specification:

| Value | Model |
| ----: | ----- |
| `0` | Classical normal linear regression |
| `1` | Cross-sectional normal-exponential stochastic frontier |
| `2` | Cross-sectional normal-half-normal stochastic frontier |
| `3` | Panel normal-exponential stochastic frontier |
| `4` | Panel normal-half-normal stochastic frontier |
| `5` | Panel Gaussian random-effects model |

## Model Estimation

The main estimation function is:

```python
model = sf_model.fit(X, y, n, T, sfa_opt, dec_crit, if_mdd)
```

It returns a results object containing parameter estimates, standard errors, model fit measures, and other diagnostics.

Depending on the selected options, the returned object may include:

- maximum likelihood estimates;
- Bayesian posterior estimates;
- Akaike information criterion;
- Bayesian information criterion;
- maximized log-likelihood;
- maximized log-posterior;
- marginal data density;
- confidence or credible intervals;
- additional summary statistics.

Display the estimation results with:

```python
model.summary()
```

Because `summary()` prints the results directly, it should not be wrapped in `print()`.

## Function Arguments

### `X`

The regressor matrix.

As the intercept is required, `X` must already contain a column of ones. It is recommended to place the intercept column as the first column:

```python
X = np.column_stack((np.ones(len(y)), regressors))
```

### `y`

The dependent variable. Its observations must be ordered consistently with the rows of `X`.

### `n`

The number of cross-sectional units.

For a cross-sectional model, this usually equals the total number of observations. For a panel model, it is the number of units.  

### `T`

The number of time periods.

For cross-sectional models:

```python
T = 1
```

For panel models, observations should be ordered by unit and then by time:

```text
unit 1: period 1, ..., period T
unit 2: period 1, ..., period T
...
unit n: period 1, ..., period T
```

### `dec_crit`

The principal model-fit criterion:

```python
dec_crit = 1
```

uses the marginal data density, whereas:

```python
dec_crit = 0
```

uses the Bayesian information criterion.

The marginal data density provides a fully Bayesian measure of model fit, while BIC is faster to compute.

### `if_mdd`

Controls whether Bayesian calculations and the marginal data density are computed:

```python
if_mdd = 1
```

enables Bayesian estimates and marginal data density calculations.

```python
if_mdd = 0
```

uses maximum likelihood estimation without calculating the marginal data density.

For a fully Bayesian analysis:

```python
dec_crit = 1
if_mdd = 1
```

For a faster estimation based on BIC:

```python
dec_crit = 0
if_mdd = 0
```

## Basic Example

```python
import numpy as np

from sfa import sf_model, inefficiency

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

model = sf_model.fit(X, y, n, T, sfa_opt, dec_crit, if_mdd)
model.summary()

# Efficiency scores based on Bayesian estimates
bayes = 1
u, te = inefficiency.jondrow(model, X, y, bayes)
```

Set:

```python
bayes = 0
```

to estimate scores using the maximum likelihood.

## Example data

Example scripts and datasets are stored in the `examples` directory on GitHub. Just copy-paste them (with the data folder) into your working directory and run there. 

A portable way to load an Excel file from `examples/data` is:

```python
import os
import pandas as pd

my_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(my_dir, 'data', 'dane.xlsx')
data = pd.read_excel(data_path, header=None, sheet_name=sf_label)
```

This works on Windows, macOS, and Linux.

## Complete Examples

The repository on GitHub includes complete examples based on simulated and empirical data, including data preparation, model estimation, result summaries, and (in)efficiency scores estimation.

See the files in:

```text
examples/
```

## Project Background and development status

This Python implementation is based on code developed for a broader MATLAB project on Bayesian model averaging in SFA. The MATLAB implementation includes the model search and Bayesian model averaging procedures, whereas this Python package currently focuses on a single-model estimation. The porject is under development. 

MATLAB package:

```text
https://github.com/KamilMakiela/SF-BMA
```

Python package:

```text
https://github.com/KamilMakiela/Python-SFA
```

## References

Jondrow, J., Lovell, C. A. K., Materov, I. S., and Schmidt, P. (1982). "*On the estimation of technical inefficiency in the stochastic frontier production function model*". Journal of Econometrics, 19(2–3), 233–238. https://doi.org/10.1016/0304-4076(82)90004-5

Makieła, K. (2026). "*Model uncertainty under non-Gaussian errors: Bayesian model averaging and selection in stochastic frontier models*". Available on arXiv.
