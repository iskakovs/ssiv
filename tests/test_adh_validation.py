"""Replication of the ShiftShareSE vignette (Autor, Dorn and Hanson data).

The ADH data are not bundled. Run ``scripts/fetch_adh.sh <dir>`` (needs R for
reading the .rda file with base R) and set ``SSIV_ADH_DIR=<dir>``; otherwise
these tests are skipped. Targets are the printed outputs of
``ShiftShareSE::reg_ss`` / ``ivreg_ss`` in the package vignette.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ssiv

ADH = os.environ.get("SSIV_ADH_DIR")
pytestmark = pytest.mark.skipif(
    not ADH or not (Path(ADH) / "adh_reg.csv").exists(), reason="set SSIV_ADH_DIR to run"
)

CTRLS = [
    "t2",
    "l_shind_manuf_cbp",
    "l_sh_popedu_c",
    "l_sh_popfborn",
    "l_sh_empl_f",
    "l_sh_routine33",
    "l_task_outsource",
    "division",
]

# method: (se, p, ci_low, ci_high), from the ShiftShareSE vignette
TARGETS = {
    "first_stage": (
        0.6310409,
        {
            "homosk": (0.02732516, 0.0, 0.5774846, 0.6845973),
            "ehw": (0.08700719, 4.083400e-13, 0.4605100, 0.8015719),
            "region_cluster": (0.09142372, 5.113909e-12, 0.4518537, 0.8102281),
            "akm": (0.05296055, 0.0, 0.5272402, 0.7348417),
            "akm0": (0.07671358, 1.282891e-03, 0.5375710, 0.8382827),
        },
    ),
    "reduced_form": (
        -0.4885687,
        {
            "homosk": (0.06332778, 1.221245e-14, -0.6126889, -0.3644485),
            "ehw": (0.11244360, 1.392685e-05, -0.7089541, -0.2681833),
            "region_cluster": (0.07578147, 1.140306e-10, -0.6370977, -0.3400398),
            "akm": (0.16419445, 2.924641e-03, -0.8103839, -0.1667535),
            "akm0": (0.25437489, 4.218033e-04, -1.2368853, -0.2397541),
        },
    ),
    "iv": (
        -0.7742267,
        {
            "homosk": (0.1069532, 4.523049e-13, -0.9838511, -0.5646022),
            "ehw": (0.1647892, 2.623532e-06, -1.0972075, -0.4512459),
            "region_cluster": (0.1758096, 1.063809e-05, -1.1188071, -0.4296462),
            "akm": (0.2403730, 1.277718e-03, -1.2453492, -0.3031041),
            "akm0": (0.3318966, 4.218033e-04, -1.6903240, -0.3893132),
        },
    ),
}


@pytest.fixture(scope="module")
def adh():
    d = Path(ADH)
    reg = pd.read_csv(d / "adh_reg.csv")
    shares = pd.read_csv(d / "adh_W.csv")
    sic = pd.read_csv(d / "adh_sic.csv")["sic"].to_numpy()
    common = dict(
        shares=shares,
        controls=CTRLS,
        categorical=["division"],
        weights="weights",
        region_cluster="statefip",
        sector_cluster=np.floor(sic / 10),
    )
    return reg, common


def _results(adh):
    reg, common = adh
    return {
        "first_stage": ssiv.reg_ss(reg, y="shock", x="IV", **common),
        "reduced_form": ssiv.reg_ss(reg, y="d_sh_empl", x="IV", **common),
        "iv": ssiv.ivreg_ss(reg, y="d_sh_empl", endog="shock", instrument="IV", **common),
    }


@pytest.mark.parametrize("model", list(TARGETS))
def test_matches_shiftshare_se(adh, model):
    res = _results(adh)[model]
    beta, rows = TARGETS[model]
    assert res.beta == pytest.approx(beta, rel=1e-6)
    for method, (se, p, lo, hi) in rows.items():
        got = res.table.loc[method]
        assert got.se == pytest.approx(se, rel=1e-6), method
        assert got.ci_low == pytest.approx(lo, rel=1e-6, abs=1e-7), method
        assert got.ci_high == pytest.approx(hi, rel=1e-6, abs=1e-7), method
        # Very small p-values differ only by floating-point noise in the tails.
        assert got.p == pytest.approx(p, rel=1e-5, abs=1e-12), method


def test_rotemberg_and_shock_level_reproduce_the_iv_estimate(adh):
    reg, common = adh
    shocks = ssiv.recover_shocks(common["shares"], reg["IV"], weights=reg["weights"])
    kw = {k: common[k] for k in ("shares", "controls", "categorical", "weights")}
    rw = ssiv.rotemberg(reg, y="d_sh_empl", endog="shock", shocks=shocks, **kw)
    sl = ssiv.shock_level_iv(reg, y="d_sh_empl", endog="shock", shocks=shocks, **kw)
    iv = ssiv.ivreg_ss(
        reg,
        y="d_sh_empl",
        endog="shock",
        instrument=ssiv.bartik(common["shares"], shocks).to_numpy(),
        method="ehw",
        **kw,
    )
    assert rw.beta_bartik == pytest.approx(iv.beta, rel=1e-9)
    assert sl.beta == pytest.approx(iv.beta, rel=1e-9)
    assert iv.beta == pytest.approx(-0.7742267, rel=1e-4)
