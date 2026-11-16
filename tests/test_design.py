import numpy as np
import pandas as pd
import pytest

import ssiv
from ssiv._design import controls_matrix


def test_object_and_bool_controls_become_dummies(sim):
    data, sh, g, cl = sim
    data = data.assign(regname=data.region.map({0: "a", 1: "b", 2: "c", 3: "d"}), flag=data.c1 > 0)
    zdf = controls_matrix(data, ["c1", "regname", "flag"], None, intercept=True)
    assert [c for c in zdf.columns if c.startswith("regname")] == [
        "regname_b",
        "regname_c",
        "regname_d",
    ]
    assert set(zdf["flag"].unique()) <= {0.0, 1.0}
    a = ssiv.reg_ss(data, y="y", x="z", shares=sh, controls=["c1", "regname"], method="ehw")
    b = ssiv.reg_ss(
        data,
        y="y",
        x="z",
        shares=sh,
        controls=["c1", "region"],
        categorical=["region"],
        method="ehw",
    )
    assert a.beta == pytest.approx(b.beta)


def test_categorical_not_in_controls_is_added(sim):
    data, sh, g, cl = sim
    a = ssiv.reg_ss(
        data, y="y", x="z", shares=sh, controls=["c1"], categorical=["region"], method="ehw"
    )
    b = ssiv.reg_ss(
        data,
        y="y",
        x="z",
        shares=sh,
        controls=["c1", "region"],
        categorical=["region"],
        method="ehw",
    )
    assert a.beta == pytest.approx(b.beta)


def test_no_intercept_and_no_controls(sim):
    data, sh, g, cl = sim
    r = ssiv.reg_ss(data, y="y", x="z", shares=sh, intercept=False, method="ehw")
    assert r.beta == pytest.approx(np.sum(data.z * data.y) / np.sum(data.z**2))


def test_shares_aligned_by_index(sim):
    data, sh, g, cl = sim
    data = data.set_index(pd.Index([f"r{i}" for i in range(len(data))]))
    shuffled = sh.set_axis(data.index).sample(frac=1.0, random_state=0)
    a = ssiv.reg_ss(data, y="y", x="z", shares=shuffled, method="akm")
    b = ssiv.reg_ss(data, y="y", x="z", shares=sh.to_numpy(), method="akm")
    assert a.se["akm"] == pytest.approx(b.se["akm"])


def test_input_errors(sim):
    data, sh, g, cl = sim
    with pytest.raises(TypeError):
        ssiv.reg_ss(data.to_numpy(), y="y", x="z", shares=sh)
    with pytest.raises(ValueError, match="non-negative"):
        ssiv.reg_ss(data.assign(w=-1.0), y="y", x="z", shares=sh, weights="w", method="ehw")
    with pytest.raises(ValueError, match="one value per row"):
        ssiv.reg_ss(data, y=np.ones(3), x="z", shares=sh, method="ehw")
    with pytest.raises(KeyError):
        ssiv.reg_ss(data, y="y", x="z", shares=sh, controls=["nope"], method="ehw")
    with pytest.raises(KeyError):
        ssiv.reg_ss(data, y="y", x="z", shares=sh, categorical=["nope"], method="ehw")
    with pytest.raises(ValueError):
        ssiv.reg_ss(data, y="y", x="z", shares=np.ones(len(data)), method="ehw")
    with pytest.raises(ValueError, match="no variation"):
        ssiv.reg_ss(data.assign(k=1.0), y="y", x="k", shares=sh, method="ehw")


def test_zero_weight_rows_are_dropped(sim):
    data, sh, g, cl = sim
    w = data.w.copy()
    w.iloc[:10] = 0.0
    a = ssiv.reg_ss(data.assign(w=w), y="y", x="z", shares=sh, weights="w", method="ehw")
    assert a.n == len(data) - 10
