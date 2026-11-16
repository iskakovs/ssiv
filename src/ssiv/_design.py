"""Assemble arrays from a DataFrame, a share matrix and options."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series | pd.DataFrame | Sequence[float]


@dataclass
class Design:
    """Arrays used by the estimators, after dropping incomplete rows."""

    y: np.ndarray
    x: np.ndarray | None  # shift-share variable (regressor or instrument)
    endog: np.ndarray | None  # endogenous regressor (IV only)
    z: np.ndarray  # controls (incl. intercept)
    w: np.ndarray  # regression weights
    shares: np.ndarray  # n x S
    sector_names: list
    control_names: list[str]
    region_cluster: np.ndarray | None
    sector_cluster: np.ndarray  # length S
    n_dropped: int
    rows: np.ndarray  # boolean mask of the rows of data that were used

    @property
    def n(self) -> int:
        return self.y.size


def _column(data: pd.DataFrame, spec, name: str) -> pd.Series:
    if spec is None:
        raise ValueError(f"{name} is required")
    if isinstance(spec, str):
        if spec not in data.columns:
            raise KeyError(f"column '{spec}' ({name}) not found in data")
        return data[spec]
    arr = np.asarray(spec)
    if arr.ndim != 1 or arr.shape[0] != len(data):
        raise ValueError(f"{name} must have one value per row of data")
    return pd.Series(arr, index=data.index)


def _shares_frame(shares, data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(shares, pd.DataFrame):
        if shares.shape[0] != len(data):
            raise ValueError("shares must have one row per row of data")
        if not shares.index.equals(data.index):
            if set(shares.index) == set(data.index):
                shares = shares.loc[data.index]
            else:
                shares = shares.set_axis(data.index, axis=0)
        return shares.astype(float)
    arr = np.asarray(shares, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != len(data):
        raise ValueError("shares must be a 2-D array with one row per row of data")
    return pd.DataFrame(arr, index=data.index)


def _is_categorical(col: pd.Series) -> bool:
    """Text and category columns become dummies; numbers and booleans enter as values."""
    return not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col))


def controls_matrix(
    data: pd.DataFrame,
    controls: Sequence[str] | None,
    categorical: Sequence[str] | None,
    intercept: bool,
) -> pd.DataFrame:
    """Numeric control matrix: intercept, numeric controls and dummies of categoricals.

    Boolean columns become 0/1. Text and category columns, and any column listed
    in ``categorical``, become dummies with the first level dropped.
    """
    controls = list(controls or [])
    categorical = list(categorical or [])
    parts = []
    if intercept:
        parts.append(pd.Series(1.0, index=data.index, name="(Intercept)"))
    for c in controls:
        if c not in data.columns:
            raise KeyError(f"control '{c}' not found in data")
        col = data[c]
        if c in categorical or _is_categorical(col):
            parts.append(
                pd.get_dummies(col, prefix=c, drop_first=intercept, dummy_na=False).astype(float)
            )
        else:
            parts.append(col.astype(float).rename(c))
    for c in categorical:
        if c not in controls:
            if c not in data.columns:
                raise KeyError(f"categorical '{c}' not found in data")
            parts.append(pd.get_dummies(data[c], prefix=c, drop_first=intercept).astype(float))
    if not parts:
        return pd.DataFrame(index=data.index)
    return pd.concat(parts, axis=1)


def build(
    data: pd.DataFrame,
    *,
    y,
    shares,
    x=None,
    endog=None,
    controls: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    intercept: bool = True,
    weights=None,
    region_cluster=None,
    sector_cluster: ArrayLike | None = None,
) -> Design:
    """Collect and align all inputs; drop rows with missing values."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    sh = _shares_frame(shares, data)
    cols = {"y": _column(data, y, "y")}
    if x is not None:
        cols["x"] = _column(data, x, "x")
    if endog is not None:
        cols["endog"] = _column(data, endog, "endog")
    cols["w"] = (
        pd.Series(1.0, index=data.index) if weights is None else _column(data, weights, "weights")
    )
    if region_cluster is not None:
        cols["region_cluster"] = _column(data, region_cluster, "region_cluster")
    zdf = controls_matrix(data, controls, categorical, intercept)

    frame = pd.DataFrame({k: v.to_numpy() for k, v in cols.items()}, index=data.index)
    numeric = [k for k in frame.columns if k != "region_cluster"]
    ok = frame[numeric].notna().all(axis=1).to_numpy().copy()  # writable under copy-on-write
    ok &= zdf.notna().all(axis=1).to_numpy()
    ok &= sh.notna().all(axis=1).to_numpy()
    if "region_cluster" in frame:
        ok &= frame["region_cluster"].notna().to_numpy()
    w = frame["w"].to_numpy(dtype=float)
    if np.any(w[ok] < 0):
        raise ValueError("weights must be non-negative")
    ok &= w > 0

    s_count = sh.shape[1]
    if sector_cluster is None:
        sc = np.arange(s_count)
    else:
        sc = np.asarray(sector_cluster)
        if sc.shape[0] != s_count:
            raise ValueError("sector_cluster must have one entry per column of shares")

    get = lambda k: frame.loc[ok, k].to_numpy(dtype=float) if k in frame else None  # noqa: E731
    return Design(
        y=get("y"),
        x=get("x"),
        endog=get("endog"),
        z=zdf.loc[ok].to_numpy(dtype=float),
        w=w[ok],
        shares=sh.loc[ok].to_numpy(dtype=float),
        sector_names=list(sh.columns),
        control_names=list(zdf.columns),
        region_cluster=frame.loc[ok, "region_cluster"].to_numpy()
        if "region_cluster" in frame
        else None,
        sector_cluster=sc,
        n_dropped=int((~ok).sum()),
        rows=ok,
    )
