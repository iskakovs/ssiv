import numpy as np
import pandas as pd
import pytest
from conftest import simulate

import ssiv


def test_rotemberg_identity(sim):
    data, sh, g, cl = sim
    kw = dict(shares=sh, controls=["c1", "region"], categorical=["region"], weights="w")
    rw = ssiv.rotemberg(data, y="y", endog="x", shocks=g, **kw)
    iv = ssiv.ivreg_ss(data, y="y", endog="x", instrument="z", method="ehw", **kw)
    t = rw.table
    assert t["alpha"].sum() == pytest.approx(1.0)
    assert (t["alpha"] * t["beta"]).sum() == pytest.approx(rw.beta_bartik)
    assert rw.beta_bartik == pytest.approx(iv.beta)


def test_rotemberg_summary_and_top(sim):
    data, sh, g, cl = sim
    rw = ssiv.rotemberg(data, y="y", endog="x", shares=sh, shocks=g)
    s = rw.summary()
    assert s["sum_positive_alpha"] + s["sum_negative_alpha"] == pytest.approx(1.0)
    top = rw.top(3)
    assert len(top) == 3
    assert top["alpha"].abs().is_monotonic_decreasing
    assert 0 < s["share_top5_abs_alpha"] <= 1


def test_rotemberg_aligns_shocks_by_label(sim):
    data, sh, g, cl = sim
    a = ssiv.rotemberg(data, y="y", endog="x", shares=sh, shocks=g)
    b = ssiv.rotemberg(data, y="y", endog="x", shares=sh, shocks=g.iloc[::-1])
    assert a.beta_bartik == pytest.approx(b.beta_bartik)


def test_shock_level_equivalence_without_constant(sim):
    data, sh, g, cl = sim
    kw = dict(shares=sh, controls=["c1", "region"], categorical=["region"], weights="w")
    sl = ssiv.shock_level_iv(data, y="y", endog="x", shocks=g, **kw)
    iv = ssiv.ivreg_ss(data, y="y", endog="x", instrument="z", method="ehw", **kw)
    assert sl.beta == pytest.approx(iv.beta, rel=1e-10)


def test_shock_level_equivalence_with_constant_and_share_sum(sim):
    data, sh, g, cl = sim
    data = data.assign(S=ssiv.share_sum(sh).to_numpy())
    kw = dict(shares=sh, controls=["c1", "S"], weights="w")
    sl = ssiv.shock_level_iv(data, y="y", endog="x", shocks=g, shock_constant=True, **kw)
    iv = ssiv.ivreg_ss(data, y="y", endog="x", instrument="z", method="ehw", **kw)
    assert sl.beta == pytest.approx(iv.beta, rel=1e-10)


def test_shock_level_se_formula(sim):
    data, sh, g, cl = sim
    sl = ssiv.shock_level_iv(data, y="y", endog="x", shares=sh, shocks=g)
    # Without controls or constant: se = sqrt(sum_n (g_n R_n)^2) / sum_l e_l z_l x_l,
    # where R_n = sum_l e_l s_ln eps_l (same structure as AKM with X_hat = g).
    yd = data.y - data.y.mean()
    xd = data.x - data.x.mean()
    z = data.z.to_numpy()
    beta = np.sum(z * yd) / np.sum(z * xd)
    eps = yd - beta * xd
    r = sh.to_numpy().T @ eps.to_numpy()
    se = np.sqrt(np.sum((g.to_numpy() * r) ** 2)) / abs(np.sum(z * xd))
    assert sl.beta == pytest.approx(beta)
    assert sl.se == pytest.approx(se, rel=1e-10)


def test_shock_level_clustering_and_diagnostics(sim):
    data, sh, g, cl = sim
    a = ssiv.shock_level_iv(data, y="y", endog="x", shares=sh, shocks=g)
    b = ssiv.shock_level_iv(data, y="y", endog="x", shares=sh, shocks=g, sector_cluster=cl)
    assert a.beta == pytest.approx(b.beta) and a.se != b.se
    assert 1 <= a.effective_n_shocks <= a.n_shocks == sh.shape[1]
    assert 0 < a.largest_weight < 1
    assert a.first_stage_f > 10
    assert set(a.data.columns) == {"s_n", "y", "x", "g"}
    assert "Shock-level IV" in repr(a)


def test_aggregate(sim):
    data, sh, g, cl = sim
    agg = ssiv.aggregate(data, shares=sh, variables=["y", "x"], weights="w")
    w = data.w.to_numpy()
    assert agg["s_n"].sum() == pytest.approx(np.sum(w * sh.sum(axis=1)))
    yd = data.y - np.average(data.y, weights=w)
    manual = (sh.to_numpy().T @ (w * yd.to_numpy())) / (sh.to_numpy().T @ w)
    np.testing.assert_allclose(agg["y"].to_numpy(), manual)
    holes = data.copy()
    holes.loc[0, "x"] = np.nan
    assert ssiv.aggregate(holes, shares=sh, variables=["y", "x"])["s_n"].sum() == pytest.approx(
        sh.iloc[1:].to_numpy().sum()
    )


def test_shock_controls_and_errors(sim):
    data, sh, g, cl = sim
    q = pd.DataFrame({"grp": cl % 2}, index=sh.columns)
    data = data.assign(qz=sh.to_numpy() @ q["grp"].to_numpy())
    r = ssiv.shock_level_iv(
        data, y="y", endog="x", shares=sh, shocks=g, controls=["qz"], shock_controls=q.to_numpy()
    )
    assert np.isfinite(r.beta) and np.isfinite(r.se)
    with pytest.raises(KeyError):
        ssiv.shock_level_iv(data, y="y", endog="x", shares=sh, shocks=g.iloc[:5])
    with pytest.raises(ValueError):
        ssiv.shock_level_iv(data, y="y", endog="x", shares=sh, shocks=g.to_numpy()[:5])
    with pytest.raises(ValueError):
        ssiv.shock_level_iv(data, y="y", endog="x", shares=sh, shocks=g, shock_controls=np.ones(3))


def test_equal_exposure_effective_number():
    data, sh, g, cl = simulate(seed=4)
    eq = pd.DataFrame(np.full(sh.shape, 1.0 / sh.shape[1]), columns=sh.columns)
    data = data.assign(z=eq.to_numpy() @ g.to_numpy())
    r = ssiv.aggregate(data, shares=eq, variables=["y"])
    wn = r["s_n"] / r["s_n"].sum()
    assert 1 / np.sum(wn**2) == pytest.approx(sh.shape[1])
