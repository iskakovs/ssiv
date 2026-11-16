"""Rotemberg weights (Goldsmith-Pinkham, Sorkin and Swift, 2020)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import _design
from ._linalg import wls_residuals
from .instrument import align_shocks


@dataclass
class RotembergResult:
    """Decomposition of a Bartik IV estimate into sector-level estimates.

    ``table`` has one row per sector with the shock ``g``, the Rotemberg weight
    ``alpha`` and the just-identified estimate ``beta`` using that sector's
    share as the only instrument. By construction
    ``sum(alpha * beta) == beta_bartik`` (over sectors with a defined ``beta``).
    """

    beta_bartik: float
    table: pd.DataFrame

    @property
    def alpha(self) -> pd.Series:
        return self.table["alpha"]

    def top(self, k: int = 5) -> pd.DataFrame:
        """Sectors with the largest absolute Rotemberg weights."""
        idx = self.table["alpha"].abs().sort_values(ascending=False).index[:k]
        return self.table.loc[idx]

    def summary(self) -> pd.Series:
        a = self.table["alpha"]
        b = self.table["beta"]
        ok = b.notna()
        top5 = self.top(5)
        return pd.Series(
            {
                "beta_bartik": self.beta_bartik,
                "n_sectors": int(len(a)),
                "sum_positive_alpha": float(a[a > 0].sum()),
                "sum_negative_alpha": float(a[a < 0].sum()),
                "share_top5_abs_alpha": float(top5["alpha"].abs().sum() / a.abs().sum()),
                "corr_alpha_g": float(np.corrcoef(a, self.table["g"])[0, 1])
                if len(a) > 1
                else np.nan,
                "weighted_mean_beta_positive": float(
                    (a[ok & (a > 0)] * b[ok & (a > 0)]).sum() / a[ok & (a > 0)].sum()
                )
                if (ok & (a > 0)).any()
                else np.nan,
                "weighted_mean_beta_negative": float(
                    (a[ok & (a < 0)] * b[ok & (a < 0)]).sum() / a[ok & (a < 0)].sum()
                )
                if (ok & (a < 0)).any()
                else np.nan,
            }
        )


def rotemberg(
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
) -> RotembergResult:
    """Rotemberg weights of a Bartik instrument.

    With residualisation on the controls (weighted), the Bartik IV estimate
    equals ``sum_k alpha_k beta_k`` where

    * ``alpha_k = g_k s_k' x / sum_j g_j s_j' x`` (``s_k`` = shares of sector k,
      ``x`` = residualised endogenous regressor),
    * ``beta_k = s_k' y / s_k' x`` is the just-identified IV estimate using
      the shares of sector k alone as the instrument.

    For panels, give share columns per sector x period (shocks likewise), so
    that each column is one instrument.
    """
    sh = shares if isinstance(shares, pd.DataFrame) else pd.DataFrame(np.asarray(shares, float))
    g = align_shocks(sh, shocks)
    d = _design.build(
        data,
        y=y,
        endog=endog,
        shares=sh,
        controls=controls,
        categorical=categorical,
        intercept=intercept,
        weights=weights,
    )
    w = d.w
    yd = wls_residuals(d.y, d.z, w)
    xd = wls_residuals(d.endog, d.z, w)
    sx = d.shares.T @ (w * xd)  # s_k' W x
    sy = d.shares.T @ (w * yd)
    num = g * sx
    total = float(num.sum())
    if total == 0:
        raise ValueError("the Bartik instrument has no first stage (sum of g_k s_k'x is zero)")
    alpha = num / total
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_k = np.where(sx != 0, sy / sx, np.nan)
    beta_bartik = float((g * sy).sum() / total)
    table = pd.DataFrame({"g": g, "alpha": alpha, "beta": beta_k}, index=sh.columns)
    table.index.name = "sector"
    return RotembergResult(beta_bartik=beta_bartik, table=table)
