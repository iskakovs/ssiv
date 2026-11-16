import numpy as np
import pandas as pd
import pytest


def simulate(n=400, s=40, seed=0, complete=False, pi=0.8, beta=-0.5, weak=False):
    """Synthetic shift-share data with known structure.

    Returns (data, shares DataFrame, shocks Series, sector clusters).
    """
    rng = np.random.default_rng(seed)
    raw = rng.gamma(0.4, size=(n, s))
    shares = raw / raw.sum(axis=1, keepdims=True)
    if not complete:
        shares *= rng.uniform(0.3, 0.9, size=(n, 1))  # incomplete shares
    sectors = [f"ind{j:02d}" for j in range(s)]
    g = pd.Series(rng.normal(0, 1, s), index=sectors, name="g")
    z = shares @ g.to_numpy()
    c1 = rng.normal(size=n)
    region = rng.integers(0, 4, size=n)
    u = rng.normal(size=n)
    strength = 0.02 if weak else pi
    x = strength * z + 0.3 * c1 + 0.5 * u + rng.normal(scale=0.5, size=n)
    y = beta * x + 0.2 * c1 + 0.1 * region + u + rng.normal(scale=0.5, size=n)
    data = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "c1": c1,
            "region": region,
            "state": rng.integers(0, 25, size=n),
            "w": rng.uniform(0.5, 2.0, size=n),
        }
    )
    sh = pd.DataFrame(shares, columns=sectors)
    clusters = np.repeat(np.arange(s // 5), 5)
    return data, sh, g, clusters


@pytest.fixture
def sim():
    return simulate()
