import numpy as np
import pandas as pd
import pytest

import ssiv


def test_bartik_and_alignment(sim):
    data, sh, g, cl = sim
    z = ssiv.bartik(sh, g)
    np.testing.assert_allclose(z.to_numpy(), data.z.to_numpy())
    np.testing.assert_allclose(ssiv.bartik(sh, g.iloc[::-1]).to_numpy(), data.z.to_numpy())
    np.testing.assert_allclose(ssiv.bartik(sh.to_numpy(), g.to_numpy()).to_numpy(), data.z)
    with pytest.raises(KeyError):
        ssiv.bartik(sh, g.iloc[1:])
    with pytest.raises(ValueError):
        ssiv.bartik(sh.to_numpy(), g.to_numpy()[1:])


def test_share_sum(sim):
    data, sh, g, cl = sim
    s = ssiv.share_sum(sh)
    np.testing.assert_allclose(s.to_numpy(), sh.sum(axis=1).to_numpy())
    assert (s < 1).all()


def test_recover_shocks(sim):
    data, sh, g, cl = sim
    rec = ssiv.recover_shocks(sh, data.z)
    np.testing.assert_allclose(rec.to_numpy(), g.to_numpy(), atol=1e-8)
    rec_w = ssiv.recover_shocks(sh, data.z, weights=data.w)
    np.testing.assert_allclose(rec_w.to_numpy(), g.to_numpy(), atol=1e-8)
    dup = sh.assign(copy=sh.iloc[:, 3])
    rec2 = ssiv.recover_shocks(dup, data.z)
    assert np.isnan(rec2["copy"]) and rec2.notna().sum() == sh.shape[1]


def test_recovered_shocks_reproduce_instrument(sim):
    data, sh, g, cl = sim
    rec = ssiv.recover_shocks(sh, data.z)
    pd.testing.assert_series_equal(ssiv.bartik(sh, rec), ssiv.bartik(sh, g), check_names=False)
