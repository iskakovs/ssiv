"""Shock-level estimation (Borusyak, Hull and Jaravel, 2022).

The location-level shift-share IV estimate equals an IV regression across
shocks (sectors) of exposure-weighted averages of residualised outcomes and
treatments on the shocks, weighted by total exposure ``s_n``. Inference at the
shock level is then valid under quasi-random assignment of shocks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import _design
from ._linalg import wls_residuals
from .instrument import align_shocks

_ND = NormalDist()


def aggregate(
    data: pd.DataFrame,
    *,
    shares,
    variables: dict | Sequence[str],
    controls: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    intercept: bool = True,
    weights=None,
) -> pd.DataFrame:
    """Exposure-weighted shock-level aggregates (the ``ssaggregate`` step).

    For each variable ``v`` (residualised on the controls with weights ``e_l``)
    returns ``v_n = sum_l e_l s_ln v_l / s_n`` with ``s_n = sum_l e_l s_ln``.
    The result has one row per sector, a column ``s_n`` and one column per
    variable. Sectors with zero total exposure are dropped.
    """
    items = list(variables.items()) if isinstance(variables, dict) else [(v, v) for v in variables]
    if not items:
        raise ValueError("give at least one variable to aggregate")
    cols = {}
    for name, v in items:
        col = data[v] if isinstance(v, str) else pd.Series(np.asarray(v, float), index=data.index)
        if len(col) != len(data):
            raise ValueError(f"variable '{name}' must have one value per row of data")
        cols[name] = col.astype(float)
    complete = pd.DataFrame(cols).notna().all(axis=1).to_numpy()
    # Mark rows with any missing variable through the first variable, so the
    # design drops them together with rows missing controls, weights or shares.
    first = cols[items[0][0]].where(complete)
    d = _design.build(
        data,
        y=first,
        shares=shares,
        controls=controls,
        categorical=categorical,
        intercept=intercept,
        weights=weights,
    )
    sh = d.shares
    s_n = sh.T @ d.w
    out = {"s_n": s_n}
    for name, col in cols.items():
        vd = wls_residuals(col.to_numpy()[d.rows], d.z, d.w)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = (sh.T @ (d.w * vd)) / s_n
    df = pd.DataFrame(out, index=pd.Index(d.sector_names, name="sector"))
    return df.loc[df["s_n"] > 0]


@dataclass
class ShockLevelResult:
    """Shock-level IV estimate with exposure-robust inference."""

    beta: float
    se: float
    p: float
    ci_low: float
    ci_high: float
    first_stage_coef: float
    first_stage_se: float
    first_stage_f: float
    n_shocks: int
    effective_n_shocks: float
    largest_weight: float
    alpha: float
    data: pd.DataFrame

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "beta": self.beta,
                "se": self.se,
                "p": self.p,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "first_stage_coef": self.first_stage_coef,
                "first_stage_se": self.first_stage_se,
                "first_stage_F": self.first_stage_f,
                "n_shocks": self.n_shocks,
                "effective_n_shocks": self.effective_n_shocks,
                "largest_weight": self.largest_weight,
            }
        )

    def __repr__(self) -> str:
        return "Shock-level IV (Borusyak, Hull and Jaravel)\n" + self.summary().to_string(
            float_format=lambda v: f"{v:.6g}"
        )


def shock_level_iv(
    data: pd.DataFrame,
    *,
    y,
    endog,
    shares,
    shocks,
    controls: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    intercept: bool = True,
    weights=None,
    shock_controls: pd.DataFrame | np.ndarray | None = None,
    shock_constant: bool = False,
    sector_cluster=None,
    alpha: float = 0.05,
    beta0: float = 0.0,
) -> ShockLevelResult:
    """IV regression at the level of shocks.

    Regresses exposure-weighted outcome averages on treatment averages,
    instrumented by the shocks, with weights ``s_n``. Without shock-level
    controls the point estimate is numerically identical to the
    location-level IV estimate with instrument ``shares @ shocks``.

    Parameters
    ----------
    shock_controls:
        Sector-level controls ``q_n`` (one row per column of ``shares``). For
        the equivalence to hold, include their exposure-weighted versions
        ``shares @ q`` among the location-level controls.
    shock_constant:
        Add a constant at the shock level. With incomplete shares, combine it
        with the sum of shares as a location-level control
        (:func:`ssiv.share_sum`).
    sector_cluster:
        Cluster labels for sectors (e.g. 3-digit industries); otherwise
        heteroskedasticity-robust standard errors at the shock level.

    Standard errors use no small-sample correction.
    """
    sh = shares if isinstance(shares, pd.DataFrame) else pd.DataFrame(np.asarray(shares, float))
    g_all = align_shocks(sh, shocks)
    agg = aggregate(
        data,
        shares=sh,
        variables={"y": y, "x": endog},
        controls=controls,
        categorical=categorical,
        intercept=intercept,
        weights=weights,
    )
    pos = sh.columns.get_indexer(agg.index)
    g = g_all[pos]
    s = agg["s_n"].to_numpy()
    yb, xb = agg["y"].to_numpy(), agg["x"].to_numpy()

    q_parts = []
    if shock_constant:
        q_parts.append(np.ones((len(g), 1)))
    if shock_controls is not None:
        q = np.asarray(shock_controls, float)
        q = q.reshape(-1, 1) if q.ndim == 1 else q
        if q.shape[0] != sh.shape[1]:
            raise ValueError("shock_controls must have one row per sector")
        q_parts.append(q[pos])
    q = np.hstack(q_parts) if q_parts else None

    gt = wls_residuals(g, q, s)
    yt = wls_residuals(yb, q, s)
    xt = wls_residuals(xb, q, s)
    den = float(np.sum(s * gt * xt))
    if den == 0:
        raise ValueError("no first stage at the shock level")
    beta = float(np.sum(s * gt * yt) / den)
    eps = yt - beta * xt

    clusters = np.arange(len(g)) if sector_cluster is None else np.asarray(sector_cluster)[pos]

    def csum(v):
        return pd.Series(v).groupby(clusters, sort=False).sum().to_numpy()

    se = float(np.sqrt(np.sum(csum(s * gt * eps) ** 2)) / abs(den))
    zc = _ND.inv_cdf(1 - alpha / 2)
    t = (beta - beta0) / se
    # First stage at the shock level: x_n on g_n
    sgg = float(np.sum(s * gt * gt))
    pi = float(np.sum(s * gt * xt) / sgg)
    u = xt - pi * gt
    pi_se = float(np.sqrt(np.sum(csum(s * gt * u) ** 2)) / sgg)
    wn = s / s.sum()
    agg = agg.assign(g=g)
    return ShockLevelResult(
        beta=beta,
        se=se,
        p=2 * _ND.cdf(-abs(t)),
        ci_low=beta - zc * se,
        ci_high=beta + zc * se,
        first_stage_coef=pi,
        first_stage_se=pi_se,
        first_stage_f=(pi / pi_se) ** 2,
        n_shocks=len(g),
        effective_n_shocks=float(1.0 / np.sum(wn**2)),
        largest_weight=float(wn.max()),
        alpha=alpha,
        data=agg,
    )
