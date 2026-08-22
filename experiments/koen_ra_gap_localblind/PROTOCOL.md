# KOEN_RA_GAP_V04_LOCALBLIND_20260822 — frozen protocol

Frozen before any V04 physical-gap outcome is calculated.

## Why V04 exists

V01 produced a strong `joined < uncertain < certain` ordering, but its physical localisation inherited VT/Takahashi word rectangles. V02 removed internal VT word boundaries completely, but its full-line splitter failed the physical-instrument validity gate. V03 added the pre-existing DINO connected-component minimum-area rule and failed the same gate. V02/V03 are retained as instrument-failure audits and are not evidence for or against the spacing hypothesis.

V04 changes the geometry rather than tuning the mask. It measures each target in a small local Yale raster while keeping the target boundary completely hidden from the image stage.

## Frozen question and classes

The hypothesis, target pair and class definitions are unchanged from V01–V03. Zandbergen–Landini IVTFF v3b running-text (`P`) loci only:

1. `joined`: literal plain-EVA `ra` inside a token;
2. `uncertain`: plain token ending `r`, ZL comma, next plain token starting `a`;
3. `certain`: plain token ending `r`, ZL period, next plain token starting `a`.

No alternate glyph pair may be substituted after target opening.

## Blinding / leakage firewall

As before, labels are sealed before image measurement. The image stage receives only event id, folio, locus and boundary-stripped character indices. `ra`, `r,a` and `r.a` are therefore identical inputs to the measurement code.

VT/Takahashi may provide only:
- page/line identity;
- the outer envelope of the complete physical line;
- a boundary-stripped line string for monotone line matching.

The V04 target localiser receives no VT internal word rectangle, word identity, word index, separator, topology, or target-adjacent x coordinate.

## Yale registration

Reuse the already-frozen CPU page registration from V02/V03: Yale IIIF derivative registered to the legacy page by CLAHE + SIFT + USAC_MAGSAC/RANSAC, then warped into the fixed 2× canonical page coordinate system. Registration parameters are unchanged.

## Local boundary-blind instrument

For a matched boundary-stripped line of `n` glyph positions and outer line width `W`, define the nominal pitch `p = W/n`.

For each target `r→a` at indices `(i,i+1)`:

1. Use a fixed context radius of 3 glyph positions on each side, clipped at line ends.
2. The nominal local context x-range is determined only from context character ordinals and `p`; no word-box x coordinate is consulted.
3. Search a fixed seven-value translation grid for the whole local context: `{-0.75,-0.50,-0.25,0,+0.25,+0.50,+0.75} × p`.
4. For each translated context, crop the continuous Yale line raster, apply the unchanged DINO adaptive ink mask (Gaussian adaptive threshold, block 31, C=12, 2×2 opening), then the pre-existing DINO 8-connected component admission rule `area >= 6 px`.
5. Split only that local context into its known number of boundary-stripped glyph positions using the unchanged low-ink dynamic-programming cut function.
6. Score each translation using the mean smoothed vertical ink projection at all internal cuts, normalized by the median positive column projection. Choose the lowest score; ties are broken by the smallest absolute translation, then the negative translation. The target label is unavailable during this choice.
7. The physical outcome is the distance from the rightmost retained ink in the target `r` slot to the leftmost retained ink in the adjacent `a` slot, floored at zero.
8. Primary normalization divides by the robust 5–95% local ink height. Raw registered-Yale pixels are mandatory sensitivity output.

No DINO embedding inference, Hugging Face Job or GPU is used.

## Alignment gates

Unchanged from V02/V03:
- primary boundary-stripped line similarity >= 0.45;
- target characters must map as exact consecutive `r`,`a` in the boundary-stripped VT stream;
- fixed similarity >=0.70 sensitivity;
- fixed exact boundary-stripped ZL=VT line sensitivity.

## Measurement-validity gate

Validity is evaluated before accepting any hypothesis result. V04 is `INSTRUMENT_FAILED` if any of the following occurs:

- pooled zero-gap rate >= 0.75;
- median local foreground fraction >= 0.30;
- any class retains <40% of its frozen candidate events;
- difference between highest and lowest class retention exceeds 0.15.

These thresholds are frozen before V04 outcomes. An instrument failure forces both scientific decisions to `NOT_RESOLVED`, regardless of p-value.

## Primary statistic and decision rule

Unchanged in substance from V02/V03. Within each folio containing both certain and uncertain cases, compute mean normalized gap(certain) − mean normalized gap(uncertain), weighted by `n_c*n_u/(n_c+n_u)`.

Permute labels 9,999 times within folio, preserving class counts, seed `6037`. Report effect, null mean, null SD, z, one-sided plus-one p, stratified AUC, and 5,000-folio-bootstrap 95% CI.

Primary is `RESOLVED` only if:
- instrument is `VALID`;
- p <= .05;
- effect >= 2 null SD above null mean;
- bootstrap 95% CI entirely >0;
- every finite leave-one-Davis-hand-out effect >0.

## Secondary continuum

`joined < uncertain < certain` is `RESOLVED` only if the primary passes and uncertain−joined has p<=.05 and z>=2 under the same within-folio null.

## Mandatory audits

- raw registered-Yale pixels;
- similarity >=0.70;
- exact boundary-stripped lines;
- class-specific retention;
- zero-gap rates;
- local foreground fraction;
- chosen translation distribution in glyph-pitch units;
- leave-one-folio-out;
- hand-specific and leave-one-hand-out effects;
- V01 direction/magnitude comparison labelled cross-instrument sensitivity only.

## Interpretation boundary

A positive result would show only that ZL boundary confidence predicts physical `r→a` ink separation under a target-boundary-blind Yale measurement. It would not establish linguistic wordhood, semantics, decipherment, or three discrete scribal categories.
