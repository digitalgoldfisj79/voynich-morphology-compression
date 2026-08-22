# KOEN_RA_GAP_V01_20260822 — frozen protocol

Frozen before any `r→a` physical-gap outcome is calculated.

## Question
For EVA `r→a`, are ZL uncertain spaces physically intermediate between joined forms and certain spaces, as proposed by Koen G? The primary falsification target is narrower: among physically comparable between-word cases, is a ZL certain boundary (`r.a`) wider than an uncertain boundary (`r,a`)?

## Fixed classes
Running-text (`P`) loci in Zandbergen–Landini IVTFF v3b only.

1. `joined`: literal plain-EVA `ra` inside a token;
2. `uncertain`: plain token ending `r`, ZL comma, next plain token starting `a`;
3. `certain`: plain token ending `r`, ZL period, next plain token starting `a`.

Drawing intrusions `<->`, alternative readings, unknown characters, high-ASCII/markup and non-running loci cannot define an event. No other glyph pair may be substituted after target opening.

## Physical instrument
Reuse the existing DINO baseline image primitives, not text-box distances:

- Voynichese.com coordinate rectangles locate image regions only;
- DINO adaptive ink mask: Gaussian adaptive threshold, block 31, C=12, then 2×2 morphological opening;
- DINO `_best_vertical_cuts` low-ink dynamic-programming partition locates character slots inside each word;
- the outcome is the raw raster distance from the rightmost detected ink of `r` to the leftmost detected ink of `a`, floored at zero;
- primary outcome divides pixels by the mean height of the two word boxes; raw pixels are a fixed sensitivity.

The ZL boundary class is not supplied to the image measurement.

## Alignment gate
Use the existing monotone physical-line reconstruction and require line-string similarity >=0.45, inherited from the prior geometry programme. Both target characters must map as exact `r` and `a` matches and be consecutive in the independent VT character stream. A fixed stricter >=0.70 sensitivity is reported.

## Circularity/leakage control
Voynichese.com/VT word segmentation is an independent human locator and could itself encode spacing judgement. Therefore the primary randomization stratifies jointly by `folio × VT topology` (`same_word` / `cross_word`). The primary effect can only use strata containing both ZL certain and uncertain cases. A VT-cross-word-only sensitivity is mandatory.

## Primary statistic and null
Within every mixed `folio × VT-topology` stratum compute mean normalized gap(certain) − mean normalized gap(uncertain), weighted by `n_c*n_u/(n_c+n_u)`. Permute certain/uncertain labels 9,999 times within each stratum, seed 6037, preserving exact class counts.

Report actual effect, null mean, null SD, z, one-sided plus-one p, stratified AUC, and a 5,000-replicate folio bootstrap CI.

Primary physical distinction is RESOLVED only if all hold:
- permutation p <= .05;
- effect >= 2 null SD above null mean;
- folio-bootstrap 95% CI entirely >0;
- every finite leave-one-hand-out primary effect remains >0.

## Secondary continuum claim
The three-way continuum is RESOLVED only if the primary passes and the stratified uncertain-minus-joined comparison also has p<=.05 and z>=2. Group distributions and overlap are reported regardless.

## Mandatory bounds
- raw-pixel outcome;
- line-similarity >=.70;
- VT-cross-word-only subset;
- folio-only null without VT-topology stratification;
- Davis/ZL hand-specific effects and leave-one-hand-out effects.

No result establishes linguistic wordhood, semantics, or decipherment.
