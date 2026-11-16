# Changelog

## 0.1.0 — unreleased

First public release (alpha).

* `reg_ss` and `ivreg_ss`: OLS and IV with a shift-share variable, with homoskedastic,
  EHW, region-clustered, AKM and AKM0 inference (Adão, Kolesár and Morales, 2019),
  sector clustering, regression weights, categorical controls, small-sample options and
  collinear-share handling.
* `rotemberg`: Rotemberg weights and sector-level just-identified estimates
  (Goldsmith-Pinkham, Sorkin and Swift, 2020).
* `aggregate` and `shock_level_iv`: shock-level aggregation and IV with robust or
  clustered errors, first-stage F, effective number of shocks (Borusyak, Hull and
  Jaravel, 2022).
* `bartik`, `share_sum`, `recover_shocks`: building blocks.
* Validation against the published ShiftShareSE results on the Autor, Dorn and Hanson
  data, run in CI.
