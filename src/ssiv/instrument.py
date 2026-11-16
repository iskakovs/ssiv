"""Constructing shift-share instruments and recovering shocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._linalg import independent_columns


def _as_frame(shares) -> pd.DataFrame:
    return shares if isinstance(shares, pd.DataFrame) else pd.DataFrame(np.asarray(shares, float))


def align_shocks(shares, shocks) -> np.ndarray:
    """Shock values ordered like the share columns (by label when both are labelled)."""
    sh = _as_frame(shares)
    if isinstance(shocks, pd.Series) and isinstance(shares, pd.DataFrame):
        missing = [c for c in sh.columns if c not in shocks.index]
        if missing:
            raise KeyError(f"no shock for {len(missing)} sector(s), e.g. {missing[:3]}")
        return shocks.loc[sh.columns].to_numpy(float)
    g = np.asarray(shocks, float)
    if g.ndim != 1 or g.shape[0] != sh.shape[1]:
        raise ValueError("shocks must have one value per column of shares")
    return g


def bartik(shares, shocks) -> pd.Series:
    """Shift-share variable ``z_l = sum_n s_ln g_n``.

    ``shares`` is a regions x sectors matrix (DataFrame or array); ``shocks``
    has one value per sector. If both are labelled (DataFrame columns and a
    Series index), shocks are aligned to the share columns by label.
    """
    sh = _as_frame(shares)
    g = align_shocks(shares, shocks)
    return pd.Series(sh.to_numpy(float) @ g, index=sh.index, name="bartik")


def share_sum(shares) -> pd.Series:
    """Sum of exposure shares per region.

    With incomplete shares (rows summing to less than one), Borusyak, Hull and
    Jaravel (2022) recommend including this sum as a control.
    """
    sh = _as_frame(shares)
    return pd.Series(sh.to_numpy(float).sum(axis=1), index=sh.index, name="share_sum")


def recover_shocks(shares, x, weights=None) -> pd.Series:
    """Recover sector shocks from a shift-share variable by least squares.

    Useful when a dataset ships the aggregated instrument ``x = shares @ g``
    but not the shocks themselves. Collinear sectors get ``NaN``.
    """
    sh = _as_frame(shares)
    a = sh.to_numpy(float)
    xv = np.asarray(x, float)
    w = np.ones(len(xv)) if weights is None else np.asarray(weights, float)
    sw = np.sqrt(w)
    keep = independent_columns(a * sw[:, None])
    g = np.full(a.shape[1], np.nan)
    g[keep] = np.linalg.lstsq(a[:, keep] * sw[:, None], xv * sw, rcond=None)[0]
    return pd.Series(g, index=sh.columns, name="shock")
