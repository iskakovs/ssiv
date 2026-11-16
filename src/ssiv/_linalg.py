"""Weighted least-squares helpers and collinearity detection."""

from __future__ import annotations

import numpy as np


def wls_residuals(a: np.ndarray, z: np.ndarray | None, w: np.ndarray) -> np.ndarray:
    """Residuals of a weighted least-squares regression of ``a`` on ``z``.

    ``a`` may be a vector or a matrix (each column is residualised).
    Rank-deficient ``z`` is handled through a minimum-norm solution, which
    leaves the residuals unchanged.
    """
    if z is None or z.shape[1] == 0:
        return np.array(a, dtype=float, copy=True)
    sw = np.sqrt(w)
    a2 = a if a.ndim == 2 else a[:, None]
    coef = np.linalg.lstsq(z * sw[:, None], a2 * sw[:, None], rcond=None)[0]
    r = a2 - z @ coef
    return r if a.ndim == 2 else r[:, 0]


def wls_coef(a: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares coefficients of ``a`` on ``z``."""
    sw = np.sqrt(w)
    return np.linalg.lstsq(z * sw[:, None], a * sw, rcond=None)[0]


def matrix_rank(z: np.ndarray | None) -> int:
    if z is None or z.shape[1] == 0:
        return 0
    return int(np.linalg.matrix_rank(z))


def independent_columns(a: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Boolean mask of columns kept when dropping collinear columns in order.

    A column is dropped when the norm of its projection off the span of the
    columns kept before it is at most ``tol`` times its own norm. This mirrors
    the column-dropping rule of R's ``lm`` (earlier columns are kept), so the
    same sectors are dropped when a share matrix is collinear.
    """
    n, k = a.shape
    norms = np.linalg.norm(a, axis=0)
    keep = norms > 0
    if np.linalg.matrix_rank(a[:, keep]) == keep.sum():
        return keep
    basis = np.zeros((n, 0))
    for j in range(k):
        if not keep[j]:
            continue
        v = a[:, j].astype(float)
        for _ in range(2):  # re-orthogonalise for numerical stability
            v = v - basis @ (basis.T @ v)
        nv = np.linalg.norm(v)
        if nv <= tol * norms[j]:
            keep[j] = False
        else:
            basis = np.column_stack([basis, v / nv])
    return keep
