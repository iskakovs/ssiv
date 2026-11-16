"""ssiv: shift-share (Bartik) instrumental variables in Python.

* :func:`reg_ss`, :func:`ivreg_ss` - OLS and IV with a shift-share variable and
  AKM / AKM0 exposure-robust inference (Adão, Kolesár and Morales, 2019).
* :func:`rotemberg` - Rotemberg weights (Goldsmith-Pinkham, Sorkin and Swift, 2020).
* :func:`aggregate`, :func:`shock_level_iv` - shock-level estimation
  (Borusyak, Hull and Jaravel, 2022).
* :func:`bartik`, :func:`share_sum`, :func:`recover_shocks` - building blocks.
"""

from ._version import __version__
from .inference import (
    METHODS,
    CollinearSharesWarning,
    ShiftShareResult,
    ivreg_ss,
    reg_ss,
)
from .instrument import bartik, recover_shocks, share_sum
from .rotemberg import RotembergResult, rotemberg
from .shocklevel import ShockLevelResult, aggregate, shock_level_iv

__all__ = [
    "METHODS",
    "CollinearSharesWarning",
    "RotembergResult",
    "ShiftShareResult",
    "ShockLevelResult",
    "__version__",
    "aggregate",
    "bartik",
    "ivreg_ss",
    "recover_shocks",
    "reg_ss",
    "rotemberg",
    "share_sum",
    "shock_level_iv",
]
