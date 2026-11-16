# ssiv

[![CI](https://github.com/iskakovs/ssiv/actions/workflows/ci.yml/badge.svg)](https://github.com/iskakovs/ssiv/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ssiv.svg)](https://pypi.org/project/ssiv/)
[![Python](https://img.shields.io/pypi/pyversions/ssiv.svg)](https://pypi.org/project/ssiv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/iskakovs/ssiv/blob/main/LICENSE)

**Shift-share (Bartik) instrumental variables in Python**: exposure-robust inference
(AKM / AKM0), Rotemberg weights and shock-level estimation, in one package.

> Status: alpha (v0.1.1). The API may change before 1.0.

## Why

Shift-share instruments, `z_l = Σ_n s_ln g_n`, combine local exposure shares with sector
shocks. They are used everywhere from trade and labour economics to regional and urban
studies. Since 2019, three papers have changed how they should be estimated and reported:

* **Adão, Kolesár and Morales (2019)**: conventional standard errors are too small,
  because regions with similar sector mixes have correlated errors. They propose the
  AKM and AKM0 confidence intervals.
* **Goldsmith-Pinkham, Sorkin and Swift (2020)**: the Bartik estimate is a weighted
  sum of sector-level estimates; Rotemberg weights show which sectors drive it.
* **Borusyak, Hull and Jaravel (2022)**: with quasi-random shocks, the estimate equals
  a shock-level regression, which gives exposure-robust inference and diagnostics.

These methods are implemented in R, Stata, MATLAB and Julia, but not in Python.
`ssiv` brings them to Python, implemented from the papers and validated against the R
reference implementation (see Validation below).

## Install

```bash
pip install ssiv
```

The development version from GitHub:

```bash
pip install "ssiv @ git+https://github.com/iskakovs/ssiv"
```

Python 3.11+, with `numpy` and `pandas` only.

## Quick start

```python
import numpy as np
import pandas as pd

import ssiv

# Simulated data: 500 regions, 40 sectors, incomplete shares, true effect -0.5
rng = np.random.default_rng(1)
n, s = 500, 40
shares = pd.DataFrame(
    rng.dirichlet(np.full(s, 0.1), n) * rng.uniform(0.4, 0.9, (n, 1)),
    columns=[f"sector{j}" for j in range(s)],
)
shocks = pd.Series(rng.normal(size=s), index=shares.columns)

data = pd.DataFrame(
    {
        "z": ssiv.bartik(shares, shocks),  # shift-share instrument
        "S": ssiv.share_sum(shares),  # sum of shares (control with incomplete shares)
        "c": rng.normal(size=n),
    }
)
u = rng.normal(size=n)  # unobserved confounder: OLS would be biased
data["x"] = 1.5 * data.z + 0.3 * data.c + u + rng.normal(scale=0.5, size=n)
data["y"] = -0.5 * data.x + 0.2 * data.c + u + rng.normal(scale=0.5, size=n)

# 1. IV with exposure-robust inference
res = ssiv.ivreg_ss(
    data,
    y="y",
    endog="x",
    instrument="z",
    shares=shares,
    controls=["c", "S"],
    method=["ehw", "akm", "akm0"],
)
print(res)

# 2. Which sectors drive the estimate?
rw = ssiv.rotemberg(data, y="y", endog="x", shares=shares, shocks=shocks, controls=["c", "S"])
print(rw.top(5))

# 3. Shock-level estimation (identical point estimate)
sl = ssiv.shock_level_iv(
    data, y="y", endog="x", shares=shares, shocks=shocks, controls=["c", "S"], shock_constant=True
)
print(sl)
```

## What is included

| Function | What it does | Reference |
|---|---|---|
| `reg_ss` | OLS with a shift-share regressor | AKM (2019) |
| `ivreg_ss` | IV with a shift-share instrument | AKM (2019) |
| `rotemberg` | Rotemberg weights and sector-level just-identified estimates | GPSS (2020) |
| `aggregate` | Exposure-weighted shock-level aggregates (the `ssaggregate` step) | BHJ (2022) |
| `shock_level_iv` | Shock-level IV with robust or clustered errors, first-stage F, effective number of shocks | BHJ (2022) |
| `bartik`, `share_sum`, `recover_shocks` | Build the instrument, the incomplete-shares control, recover shocks from an aggregated instrument | — |

`reg_ss` and `ivreg_ss` report five kinds of inference side by side:

| `method` | Inference |
|---|---|
| `"homosk"` | homoskedastic |
| `"ehw"` | heteroskedasticity-robust (Eicker–Huber–White) |
| `"region_cluster"` | clustered by region groups (`region_cluster=`) |
| `"akm"` | Adão–Kolesár–Morales, optionally clustering sectors (`sector_cluster=`) |
| `"akm0"` | AKM with the null imposed: test inversion, valid with weak instruments |

## Validation

`ssiv` reproduces the published output of the R package
[ShiftShareSE](https://github.com/kolesarm/ShiftShareSE) on the Autor, Dorn and Hanson
(2013) data. That covers the first stage, reduced form and IV, under all five methods:
every estimate, standard error and confidence bound matches to the 7 printed digits.

| IV: effect of import exposure on employment | ShiftShareSE (R) | ssiv |
|---|---|---|
| Estimate | −0.7742267 | −0.7742267 |
| EHW s.e. | 0.1647892 | 0.1647892 |
| AKM s.e. | 0.2403730 | 0.2403730 |
| AKM0 95% CI | [−1.6903240, −0.3893132] | [−1.6903240, −0.3893132] |

To run it yourself (needs R to read the `.rda` file):

```bash
bash scripts/fetch_adh.sh adh_data
SSIV_ADH_DIR=adh_data pytest tests/test_adh_validation.py
python examples/adh_replication.py adh_data
```

The test suite also checks exact identities:
* Rotemberg weights reproduce the Bartik estimate.
* The shock-level estimate equals the region-level one, with and without a shock-level
  constant.
* AKM0 intervals invert the AKM0 test.

## Conventions

* **Small-sample corrections** (`small_sample=`): `"stata"` divides the residual variance
  by `n − p`, uses HC1, and multiplies clustered variances by `G/(G−1)·(n−1)/(n−p)`;
  `"none"` applies none. The defaults follow ShiftShareSE: `"stata"` for `reg_ss`,
  `"none"` for `ivreg_ss`. AKM, AKM0 and shock-level errors never use a correction.
* **AKM**: sector shocks net of controls are estimated by a weighted regression of the
  residualised shift-share variable on the shares. Residual sums are aggregated by sector
  and squared within sector clusters.
* **Collinear shares**: collinear columns are dropped in order, keeping earlier columns,
  with a `CollinearSharesWarning`. This is the same rule as R's `lm`.
* **AKM0** confidence sets come from inverting the test. When identification is weak the
  set can be unbounded; it is then reported as `(-inf, inf)` with a note. Its "standard
  error" is the set's length divided by `2z`, as in ShiftShareSE.
* **Shock-level IV**: without shock-level controls, the estimate is numerically identical
  to the location-level one. With incomplete shares, add `share_sum` as a location
  control and set `shock_constant=True`.

## Roadmap

* Multiple shift-share instruments and overidentified models.
* Leave-one-out instruments and panel helpers (sector × period cells).
* Diagnostics: share balance (GPSS), shock balance and pre-trend tests (BHJ).
* Integration with `pyfixest` for high-dimensional fixed effects.

## References

* Adão, R., Kolesár, M. and Morales, E. (2019). Shift-Share Designs: Theory and
  Inference. *Quarterly Journal of Economics*, 134(4), 1949–2010.
* Borusyak, K., Hull, P. and Jaravel, X. (2022). Quasi-Experimental Shift-Share Research
  Designs. *Review of Economic Studies*, 89(1), 181–213.
* Goldsmith-Pinkham, P., Sorkin, I. and Swift, H. (2020). Bartik Instruments: What, When,
  Why, and How. *American Economic Review*, 110(8), 2586–2624.
* Autor, D., Dorn, D. and Hanson, G. (2013). The China Syndrome. *American Economic
  Review*, 103(6), 2121–2168.

## Citation

If you use `ssiv` in research, please cite it (see
[CITATION.cff](https://github.com/iskakovs/ssiv/blob/main/CITATION.cff)) together with
the papers whose methods you use.

## License

MIT © 2026 Serik Iskakov. The ADH data used for validation are not distributed with this
package; `scripts/fetch_adh.sh` downloads them from the ShiftShareSE repository.
