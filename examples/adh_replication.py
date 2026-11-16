"""Replicate the ShiftShareSE vignette: effect of Chinese import competition on
employment (Autor, Dorn and Hanson, 2013), with AKM/AKM0 inference.

First fetch the data (needs R):   bash scripts/fetch_adh.sh adh_data
Then run:                         python examples/adh_replication.py adh_data
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ssiv


def main(data_dir: str) -> None:
    d = Path(data_dir)
    reg = pd.read_csv(d / "adh_reg.csv")
    shares = pd.read_csv(d / "adh_W.csv")
    sic = pd.read_csv(d / "adh_sic.csv")["sic"].to_numpy()
    # Columns are industry x period cells: label them by SIC code and occurrence.
    codes = pd.Series(sic.astype(int).astype(str))
    shares.columns = "SIC" + codes + "_" + (codes.groupby(codes).cumcount() + 1).astype(str)

    common = dict(
        shares=shares,
        controls=[
            "t2",
            "l_shind_manuf_cbp",
            "l_sh_popedu_c",
            "l_sh_popfborn",
            "l_sh_empl_f",
            "l_sh_routine33",
            "l_task_outsource",
            "division",
        ],
        categorical=["division"],
        weights="weights",
        region_cluster="statefip",
        sector_cluster=np.floor(sic / 10),  # cluster sectors by 3-digit SIC
    )

    print(ssiv.reg_ss(reg, y="shock", x="IV", **common), "\n")  # first stage
    print(ssiv.reg_ss(reg, y="d_sh_empl", x="IV", **common), "\n")  # reduced form
    print(ssiv.ivreg_ss(reg, y="d_sh_empl", endog="shock", instrument="IV", **common), "\n")

    # The data ship the aggregated instrument only; recover the sector shocks.
    shocks = ssiv.recover_shocks(shares, reg["IV"], weights=reg["weights"])
    kw = {k: common[k] for k in ("shares", "controls", "categorical", "weights")}

    rw = ssiv.rotemberg(reg, y="d_sh_empl", endog="shock", shocks=shocks, **kw)
    print("Rotemberg weights: top 5 sectors\n", rw.top(5), "\n")
    print(rw.summary(), "\n")

    sl = ssiv.shock_level_iv(
        reg,
        y="d_sh_empl",
        endog="shock",
        shocks=shocks,
        sector_cluster=np.floor(sic / 10),
        **kw,
    )
    print(sl)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "adh_data")
