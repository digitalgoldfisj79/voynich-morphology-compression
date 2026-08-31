# CPU sensitivity parameters — frozen before feature extraction

Protocol: `2026-08-31.greek-ductus.v1`

These values operationalize diagnostics already required by `ANALYSIS_PLAN.md` and are frozen before the first Greek-vs-Voynich feature values are computed.

## Stricter binarization
Primary `extract_cpu.ink_mask` uses adaptive Gaussian thresholding with `C=17` after the same page-edge crop and 3x3 Gaussian blur.

The sole stricter-threshold sensitivity uses the **same** algorithm and block-size rule with `C=21`; all subsequent component grouping, crop geometry gates, normalization, feature extraction, aggregation and distance calculations are unchanged.

No further threshold values will be tried based on the result.

## Low-information crop exclusion
On the normalized black-on-white crop, compute the binary skeleton and local 8-neighbour degree.

Exclude a crop in the predeclared sensitivity if either:
1. skeleton endpoints <= 2, skeleton junctions = 0, and ink bounding-box aspect ratio < 0.55; or
2. hole count >= 1 and ink bounding-box aspect ratio is between 0.75 and 1.33 inclusive.

This diagnostic cannot replace the primary all-crop result.

## Repository sensitivities
Because operational acquisition spans hosts after the Bodleian failure:
- primary CPU result uses black-on-white normalized crops only;
- compute a no-Dresden sensitivity;
- compute a no-Leipzig sensitivity if every family retains at least two manuscripts; otherwise report it as underidentified rather than forcing a statistic;
- report family-by-host counts explicitly.

## Greek provenance sensitivities
- recompute after excluding Ferrara `Cod.graec. 256`;
- separately report descriptive distance to the centroid of the securely Byzantine/Greek-East pair `Mscr.Dresd.Da.61` and `Mscr.Dresd.Da.47`; this two-manuscript subset is not promoted to a new headline test.
