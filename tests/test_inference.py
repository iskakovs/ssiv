from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from conftest import simulate

import ssiv

Z = NormalDist().inv_cdf(0.975)


def common(sh, clusters, **kw):
    out = dict(
        shares=sh,
        controls=["c1", "region"],
        categorical=["region"],
        weights="w",
        region_cluster="state",
        sector_cluster=clusters,
    )
    out.update(kw)
    return out


def test_ols_point_estimate_matches_weighted_least_squares(sim):
    data, sh, g, cl = sim
    r = ssiv.reg_ss(data, y="y", x="z", **common(sh, cl))
    zmat = np.column_stack(
        [
            np.ones(len(data)),
            data.z,
            data.c1,
            pd.get_dummies(data.region, drop_first=True).astype(float),
        ]
    )
    sw = np.sqrt(data.w.to_numpy())
    coef = np.linalg.lstsq(zmat * sw[:, None], data.y.to_numpy() * sw, rcond=None)[0]
    assert r.beta == pytest.approx(coef[1])


def test_iv_with_instrument_as_regressor_equals_ols(sim):
    data, sh, g, cl = sim
    ols = ssiv.reg_ss(data, y="y", x="z", small_sample="none", **common(sh, cl))
    iv = ssiv.ivreg_ss(data, y="y", endog="z", instrument="z", **common(sh, cl))
    assert iv.beta == pytest.approx(ols.beta)
    pd.testing.assert_frame_equal(iv.table, ols.table)


def test_normal_intervals_are_symmetric(sim):
    data, sh, g, cl = sim
    r = ssiv.ivreg_ss(data, y="y", endog="x", instrument="z", **common(sh, cl))
    for m in ["homosk", "ehw", "region_cluster", "akm"]:
        row = r.table.loc[m]
        assert row.ci_low == pytest.approx(r.beta - Z * row.se)
        assert row.ci_high == pytest.approx(r.beta + Z * row.se)


def test_akm_matches_closed_form_with_complete_shares():
    # With complete shares and only an intercept, residualised X equals
    # shares @ (g - c), so the estimated shocks are g minus a constant.
    data, sh, g, cl = simulate(complete=True, seed=3)
    r = ssiv.reg_ss(data, y="y", x="z", shares=sh, weights="w", method="akm")
    w = data.w.to_numpy()
    xd = data.z - np.average(data.z, weights=w)
    yd = data.y - np.average(data.y, weights=w)
    beta = np.sum(w * xd * yd) / np.sum(w * xd**2)
    e = yd - beta * xd
    gc = g.to_numpy() - np.average(data.z, weights=w)
    rs = sh.to_numpy().T @ (w * e)
    se = np.sqrt(np.sum((gc * rs) ** 2)) / np.sum(w * xd**2)
    assert r.table.loc["akm", "se"] == pytest.approx(se, rel=1e-10)


def test_akm0_set_contains_estimate_and_inverts_the_test(sim):
    data, sh, g, cl = sim
    r = ssiv.ivreg_ss(data, y="y", endog="x", instrument="z", **common(sh, cl))
    lo, hi = r.table.loc["akm0", ["ci_low", "ci_high"]]
    assert lo < r.beta < hi
    for bound in (lo, hi):
        rb = ssiv.ivreg_ss(
            data, y="y", endog="x", instrument="z", beta0=bound, method="akm0", **common(sh, cl)
        )
        assert rb.table.loc["akm0", "p"] == pytest.approx(0.05, abs=1e-8)


def test_akm0_p_value_of_iv_equals_reduced_form(sim):
    data, sh, g, cl = sim
    iv = ssiv.ivreg_ss(data, y="y", endog="x", instrument="z", method="akm0", **common(sh, cl))
    rf = ssiv.reg_ss(data, y="y", x="z", method="akm0", **common(sh, cl))
    assert iv.table.loc["akm0", "p"] == pytest.approx(rf.table.loc["akm0", "p"])


def test_weak_instrument_gives_unbounded_akm0_set():
    data, sh, g, cl = simulate(weak=True, seed=11, n=150)
    r = ssiv.ivreg_ss(data, y="y", endog="x", instrument="z", shares=sh, method=["akm", "akm0"])
    assert np.isinf(r.table.loc["akm0", "ci_high"]) or np.isinf(r.table.loc["akm0", "ci_low"])
    assert any("unbounded" in n for n in r.notes)


def test_small_sample_factors(sim):
    data, sh, g, cl = sim
    a = ssiv.reg_ss(data, y="y", x="z", small_sample="stata", **common(sh, cl))
    b = ssiv.reg_ss(data, y="y", x="z", small_sample="none", **common(sh, cl))
    n, p = len(data), 1 + 1 + 1 + 3  # intercept, c1, 3 region dummies, regressor
    g_ = data.state.nunique()
    assert (a.se["homosk"] / b.se["homosk"]) ** 2 == pytest.approx(n / (n - p))
    assert (a.se["ehw"] / b.se["ehw"]) ** 2 == pytest.approx(n / (n - p))
    assert (a.se["region_cluster"] / b.se["region_cluster"]) ** 2 == pytest.approx(
        g_ / (g_ - 1) * (n - 1) / (n - p)
    )
    assert a.se["akm"] == b.se["akm"] and a.se["akm0"] == b.se["akm0"]


def test_unit_weights_equal_no_weights(sim):
    data, sh, g, cl = sim
    data = data.assign(one=1.0)
    a = ssiv.ivreg_ss(
        data, y="y", endog="x", instrument="z", shares=sh, weights="one", method=["ehw", "akm"]
    )
    b = ssiv.ivreg_ss(data, y="y", endog="x", instrument="z", shares=sh, method=["ehw", "akm"])
    pd.testing.assert_frame_equal(a.table, b.table)


def test_singleton_sector_clusters_equal_no_clustering(sim):
    data, sh, g, cl = sim
    a = ssiv.reg_ss(data, y="y", x="z", shares=sh, method="akm")
    b = ssiv.reg_ss(
        data, y="y", x="z", shares=sh, method="akm", sector_cluster=np.arange(sh.shape[1])
    )
    assert a.se["akm"] == pytest.approx(b.se["akm"])


def test_collinear_shares_are_dropped_with_warning(sim):
    data, sh, g, cl = sim
    base = ssiv.reg_ss(data, y="y", x="z", shares=sh, method=["akm", "akm0"])
    dup = sh.assign(dup=sh.iloc[:, 0] * 2.0)
    with pytest.warns(ssiv.CollinearSharesWarning):
        r = ssiv.reg_ss(data, y="y", x="z", shares=dup, method=["akm", "akm0"])
    assert r.n_sectors_dropped == 1
    pd.testing.assert_frame_equal(r.table, base.table)


def test_missing_rows_are_dropped(sim):
    data, sh, g, cl = sim
    full = ssiv.reg_ss(data.iloc[5:], y="y", x="z", shares=sh.iloc[5:], method="ehw")
    holes = data.copy()
    holes.loc[:4, "y"] = np.nan
    r = ssiv.reg_ss(holes, y="y", x="z", shares=sh, method="ehw")
    assert r.n == len(data) - 5
    assert r.beta == pytest.approx(full.beta)


def test_array_inputs(sim):
    data, sh, g, cl = sim
    a = ssiv.reg_ss(data, y="y", x="z", shares=sh, method="akm")
    b = ssiv.reg_ss(
        data, y=data.y.to_numpy(), x=data.z.to_numpy(), shares=sh.to_numpy(), method="akm"
    )
    assert a.beta == pytest.approx(b.beta) and a.se["akm"] == pytest.approx(b.se["akm"])


def test_errors(sim):
    data, sh, g, cl = sim
    with pytest.raises(ValueError, match="region_cluster"):
        ssiv.reg_ss(data, y="y", x="z", shares=sh, method="region_cluster")
    with pytest.raises(ValueError, match="unknown method"):
        ssiv.reg_ss(data, y="y", x="z", shares=sh, method="bootstrap")
    with pytest.raises(ValueError):
        ssiv.reg_ss(data, y="y", x="z", shares=sh.iloc[:10])
    with pytest.raises(KeyError):
        ssiv.reg_ss(data, y="nope", x="z", shares=sh)
    with pytest.raises(ValueError, match="sector_cluster"):
        ssiv.reg_ss(data, y="y", x="z", shares=sh, sector_cluster=[1, 2])
    with pytest.raises(ValueError, match="small_sample"):
        ssiv.reg_ss(data, y="y", x="z", shares=sh, small_sample="hc3")


def test_summary_text(sim):
    data, sh, g, cl = sim
    r = ssiv.ivreg_ss(data, y="y", endog="x", instrument="z", **common(sh, cl))
    text = r.summary()
    assert "AKM0" in text and "Instrument: z" in text and "Estimate:" in text
    assert list(r.conf_int().columns) == ["ci_low", "ci_high"]
