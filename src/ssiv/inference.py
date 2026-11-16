"""OLS and IV regressions with a shift-share variable, with exposure-robust inference.

Implements the AKM and AKM0 confidence intervals of Adão, Kolesár and Morales
(2019, *Quarterly Journal of Economics* 134(4)), alongside the usual
homoskedastic, heteroskedasticity-robust (EHW) and region-clustered errors.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

from . import _design
from ._linalg import independent_columns, matrix_rank, wls_coef, wls_residuals

METHODS = ("homosk", "ehw", "region_cluster", "akm", "akm0")
LABELS = {
    "homosk": "Homoscedastic",
    "ehw": "EHW",
    "region_cluster": "Reg. cluster",
    "akm": "AKM",
    "akm0": "AKM0",
}
SmallSample = Literal["stata", "none"]
_ND = NormalDist()


class CollinearSharesWarning(UserWarning):
    """Some columns of the share matrix were collinear and were dropped."""


@dataclass
class ShiftShareResult:
    """Point estimate and inference under several methods.

    ``table`` has one row per method with columns ``se``, ``p``, ``ci_low`` and
    ``ci_high``. For AKM0 the confidence set comes from test inversion; ``se``
    is its length divided by ``2 z`` (as in ShiftShareSE) and may be infinite
    when the set is unbounded.
    """

    kind: str
    beta: float
    table: pd.DataFrame
    n: int
    n_sectors: int
    n_sectors_dropped: int
    alpha: float
    beta0: float
    small_sample: str
    y: str
    regressor: str
    instrument: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def se(self) -> pd.Series:
        return self.table["se"]

    @property
    def pvalues(self) -> pd.Series:
        return self.table["p"]

    def conf_int(self) -> pd.DataFrame:
        return self.table[["ci_low", "ci_high"]]

    def summary(self) -> str:
        lvl = int(round((1 - self.alpha) * 100))
        head = (
            f"{'IV' if self.kind == 'iv' else 'OLS'} with a shift-share "
            f"{'instrument' if self.kind == 'iv' else 'regressor'}\n"
            f"Outcome: {self.y}   Regressor: {self.regressor}"
            + (f"   Instrument: {self.instrument}" if self.instrument else "")
            + f"\nObservations: {self.n}   Sectors: {self.n_sectors}"
            + (f" ({self.n_sectors_dropped} collinear dropped)" if self.n_sectors_dropped else "")
            + f"\nEstimate: {self.beta:.7g}\n\n"
        )
        t = self.table.rename(index=LABELS).rename(
            columns={
                "se": "Std. Error",
                "p": f"p-value (H0: beta={self.beta0:g})",
                "ci_low": f"Lower {lvl}% CI",
                "ci_high": f"Upper {lvl}% CI",
            }
        )
        return head + t.to_string(float_format=lambda v: f"{v:.7g}")

    def __repr__(self) -> str:
        return self.summary()


def _methods(method) -> list[str]:
    if method == "all":
        return list(METHODS)
    ms = [method] if isinstance(method, str) else list(method)
    bad = set(ms) - set(METHODS)
    if bad:
        raise ValueError(f"unknown method(s) {sorted(bad)}; choose from {METHODS} or 'all'")
    return [m for m in METHODS if m in ms]


def _cluster_sums(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return pd.Series(values).groupby(groups, sort=False).sum().to_numpy()


def _quadratic_set(a2: float, a1: float, a0: float) -> tuple[float, float]:
    """Set {b : a2 b^2 + a1 b + a0 <= 0}, returned as (low, high).

    Unbounded or disjoint sets are returned as (-inf, inf).
    """
    disc = a1 * a1 - 4 * a2 * a0
    if a2 > 0 and disc >= 0:
        r = np.sqrt(disc)
        lo, hi = (-a1 - r) / (2 * a2), (-a1 + r) / (2 * a2)
        return float(min(lo, hi)), float(max(lo, hi))
    return float("-inf"), float("inf")


def _fit(
    d: _design.Design,
    *,
    iv: bool,
    method,
    alpha: float,
    beta0: float,
    small_sample: SmallSample,
    names: dict,
) -> ShiftShareResult:
    if small_sample not in ("stata", "none"):
        raise ValueError("small_sample must be 'stata' or 'none'")
    ms = _methods(method)
    if "region_cluster" in ms and d.region_cluster is None:
        raise ValueError("method 'region_cluster' needs region_cluster=")
    w, z = d.w, d.z
    n = d.n
    xd = wls_residuals(d.x, z, w)
    yd = wls_residuals(d.y, z, w)
    rd = wls_residuals(d.endog, z, w) if iv else xd
    den = float(np.sum(w * xd * rd))
    scale = max(float(np.sum(w * d.x**2)), np.finfo(float).tiny)
    if float(np.sum(w * xd**2)) <= 1e-12 * scale or den == 0:
        raise ValueError("the shift-share variable has no variation after partialling out controls")
    beta = float(np.sum(w * xd * yd) / den)
    e = yd - beta * rd
    p_reg = matrix_rank(z) + 1
    stata = small_sample == "stata"
    zcrit = _ND.inv_cdf(1 - alpha / 2)
    notes: list[str] = []

    rows = {}

    def normal_row(var: float) -> dict:
        se = float(np.sqrt(var))
        t = (beta - beta0) / se if se > 0 else np.inf
        return dict(
            se=se, p=2 * _ND.cdf(-abs(t)), ci_low=beta - zcrit * se, ci_high=beta + zcrit * se
        )

    if "homosk" in ms:
        sigma2 = float(np.sum(w * e**2)) / (n - p_reg if stata else n)
        rows["homosk"] = normal_row(sigma2 * float(np.sum(w * xd**2)) / den**2)
    if "ehw" in ms:
        v = float(np.sum((w * xd * e) ** 2)) / den**2
        rows["ehw"] = normal_row(v * (n / (n - p_reg) if stata else 1.0))
    if "region_cluster" in ms:
        g = d.region_cluster
        n_g = len(pd.unique(g))
        v = float(np.sum(_cluster_sums(w * xd * e, g) ** 2)) / den**2
        corr = n_g / (n_g - 1) * (n - 1) / (n - p_reg) if stata else 1.0
        rows["region_cluster"] = normal_row(v * corr)

    n_dropped = 0
    if "akm" in ms or "akm0" in ms:
        sw = np.sqrt(w)
        keep = independent_columns(d.shares * sw[:, None])
        n_dropped = int((~keep).sum())
        if n_dropped:
            warnings.warn(
                f"share matrix is collinear: dropping {n_dropped} sector(s) for AKM inference "
                "(earlier columns are kept; reordering columns can change which are dropped)",
                CollinearSharesWarning,
                stacklevel=3,
            )
            notes.append(f"{n_dropped} collinear sectors dropped for AKM/AKM0")
        sh = d.shares[:, keep]
        clusters = d.sector_cluster[keep]
        # Shocks net of controls: weighted regression of residualised X on raw shares.
        xs_hat = wls_coef(xd, sh, w)

        if "akm" in ms:
            r = sh.T @ (w * e)
            a = _cluster_sums(xs_hat * r, clusters)
            rows["akm"] = normal_row(float(np.sum(a**2)) / den**2)

        if "akm0" in ms:
            ry = sh.T @ (w * yd)
            rr = sh.T @ (w * rd)
            ac = _cluster_sums(xs_hat * ry, clusters)
            bc = _cluster_sums(xs_hat * rr, clusters)
            num_y = float(np.sum(w * xd * yd))
            # Test of H0: beta = b uses residuals under the null, y - b * r.
            # Confidence set: {b : (num_y - b den)^2 <= z^2 sum_c (a_c - b b_c)^2}
            a2 = den**2 - zcrit**2 * float(np.sum(bc**2))
            a1 = -2 * num_y * den + 2 * zcrit**2 * float(np.sum(ac * bc))
            a0 = num_y**2 - zcrit**2 * float(np.sum(ac**2))
            lo, hi = _quadratic_set(a2, a1, a0)
            if not np.isfinite(lo):
                notes.append("AKM0 confidence set is unbounded (weak identification)")
            v0 = float(np.sum((ac - beta0 * bc) ** 2))
            t0 = (num_y - beta0 * den) / np.sqrt(v0) if v0 > 0 else np.inf
            rows["akm0"] = dict(
                se=(hi - lo) / (2 * zcrit), p=2 * _ND.cdf(-abs(t0)), ci_low=lo, ci_high=hi
            )

    table = pd.DataFrame.from_dict(rows, orient="index")[["se", "p", "ci_low", "ci_high"]]
    table.index.name = "method"
    return ShiftShareResult(
        kind="iv" if iv else "ols",
        beta=beta,
        table=table.loc[[m for m in METHODS if m in rows]],
        n=n,
        n_sectors=d.shares.shape[1],
        n_sectors_dropped=n_dropped,
        alpha=alpha,
        beta0=beta0,
        small_sample=small_sample,
        notes=notes,
        **names,
    )


def _label(v, default: str) -> str:
    return v if isinstance(v, str) else default


def reg_ss(
    data: pd.DataFrame,
    *,
    y,
    x,
    shares,
    controls: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    intercept: bool = True,
    weights=None,
    region_cluster=None,
    sector_cluster=None,
    method="all",
    alpha: float = 0.05,
    beta0: float = 0.0,
    small_sample: SmallSample = "stata",
) -> ShiftShareResult:
    """OLS of ``y`` on a shift-share regressor ``x`` and controls.

    Parameters
    ----------
    data:
        One row per region (or region-period).
    y, x:
        Column names (or arrays) of the outcome and of the shift-share
        regressor ``x = shares @ shocks``.
    shares:
        Share matrix, one row per row of ``data`` and one column per sector.
    controls, categorical, intercept:
        Control variables. Columns listed in ``categorical`` (and any column of
        object or category dtype) enter as dummies.
    weights:
        Regression weights (column name or array).
    region_cluster:
        Region-level cluster variable for the ``"region_cluster"`` method.
    sector_cluster:
        One cluster label per sector (column of ``shares``) for AKM/AKM0, e.g.
        3-digit industry codes. Defaults to no clustering of sectors.
    method:
        ``"all"`` or a subset of ``("homosk", "ehw", "region_cluster", "akm", "akm0")``.
    alpha:
        Significance level of the confidence intervals.
    beta0:
        Null value for p-values (and for the AKM0 test).
    small_sample:
        ``"stata"`` applies the usual corrections (residual variance divided by
        ``n - p``, HC1, and ``G/(G-1) (n-1)/(n-p)`` for clusters); ``"none"``
        uses none. AKM and AKM0 never use a correction. The default matches
        ``ShiftShareSE::reg_ss``.
    """
    d = _design.build(
        data,
        y=y,
        x=x,
        shares=shares,
        controls=controls,
        categorical=categorical,
        intercept=intercept,
        weights=weights,
        region_cluster=region_cluster,
        sector_cluster=sector_cluster,
    )
    return _fit(
        d,
        iv=False,
        method=method,
        alpha=alpha,
        beta0=beta0,
        small_sample=small_sample,
        names=dict(y=_label(y, "y"), regressor=_label(x, "x")),
    )


def ivreg_ss(
    data: pd.DataFrame,
    *,
    y,
    endog,
    instrument,
    shares,
    controls: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    intercept: bool = True,
    weights=None,
    region_cluster=None,
    sector_cluster=None,
    method="all",
    alpha: float = 0.05,
    beta0: float = 0.0,
    small_sample: SmallSample = "none",
) -> ShiftShareResult:
    """IV regression of ``y`` on ``endog`` with a shift-share ``instrument``.

    The model is just identified: one endogenous regressor and one shift-share
    instrument ``instrument = shares @ shocks``. See :func:`reg_ss` for the
    other parameters. ``small_sample`` defaults to ``"none"``, matching
    ``ShiftShareSE::ivreg_ss``.

    AKM0 inverts an Anderson-Rubin-type test, so its confidence set remains
    valid when the instrument is weak (it may then be unbounded).
    """
    d = _design.build(
        data,
        y=y,
        x=instrument,
        endog=endog,
        shares=shares,
        controls=controls,
        categorical=categorical,
        intercept=intercept,
        weights=weights,
        region_cluster=region_cluster,
        sector_cluster=sector_cluster,
    )
    return _fit(
        d,
        iv=True,
        method=method,
        alpha=alpha,
        beta0=beta0,
        small_sample=small_sample,
        names=dict(
            y=_label(y, "y"),
            regressor=_label(endog, "endog"),
            instrument=_label(instrument, "instrument"),
        ),
    )
