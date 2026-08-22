# KOEN_RA_GAP_V03_BOUNDARYBLIND_CC_20260822 — frozen protocol

Frozen after V02 was diagnosed as a failed physical instrument and before any V03 outcome is calculated.

## Status of V01 and V02

- **V01:** numerical results are retained as an instrument-specific audit, but the description of the VT/Takahashi cross-word sensitivity as an independent-transcriber control is retracted. V01 measured inside VT word boxes.
- **V02:** the boundary-blind design was scientifically preferable, but its full-line adaptive mask failed a prespecified measurement-degeneracy audit. Across classes, 82.0–87.4% of measured gaps were zero and the inferred 5–95% line-ink height occupied roughly 89–90% of the complete crop height. This is inconsistent with a usable glyph-edge mask and indicates parchment/background/bleed-through fragments were entering the edge measurement. V02 is therefore classified **INSTRUMENT_FAILED**. Its near-null class comparison is not evidence against the spacing hypothesis.

No V02 class effect, p-value, ordering, or V01 target value is used to choose the V03 cleanup rule.

## Single corrective change

V03 keeps the entire V02 protocol and adds exactly one pre-existing DINO baseline rule before line cutting and edge measurement:

> retain connected ink components only when `area >= 6`, `width > 0`, and `height > 0` under 8-connectivity.

`minimum_area=6` is inherited verbatim from the July 2026 DINO baseline function `connected_component_proposals(rgb, word_id, minimum_area=6, merge_gap_px=3)`. It predates Koen's r→a experiment and was not selected or tuned using V01/V02 class outcomes.

No other threshold, target, class definition, alignment gate, statistic, seed, decision rule or image-processing parameter changes.

## Question and fixed classes

For EVA `r→a`, do ZL uncertain spaces occupy an intermediate physical separation regime between joined `ra` and certain `r.a` when measured on one continuous registered-Yale line image without exposing VT/Takahashi internal word boundaries to the physical instrument?

Zandbergen–Landini IVTFF v3b running-text (`P`) loci only:

1. `joined`: literal plain-EVA `ra` inside a token;
2. `uncertain`: plain token ending `r`, ZL comma, next plain token starting `a`;
3. `certain`: plain token ending `r`, ZL period, next plain token starting `a`.

The directional prediction remains `joined < uncertain < certain`.

## Leakage firewall

As in V02:

- `targets_blinded.json` contains event id, folio, locus and boundary-stripped ZL character indices, but no class label;
- `labels_sealed.json` contains event id → class and is hashed;
- the image stage writes `measurements_blind.csv` before labels are joined;
- ZL punctuation/space marks are deleted before physical segmentation, so `ra`, `r,a` and `r.a` are identical to the image instrument at the target boundary.

The implementation runs in one process, so this is an auditable data-flow blind rather than cryptographic process isolation. The physical measurement functions do not receive or branch on class labels.

## Yale physical instrument

1. Fetch the current Yale Beinecke IIIF manifest and fixed-width Yale derivatives.
2. Register legacy page raster to Yale using the existing DINO CPU registration method: SIFT + CLAHE + USAC_MAGSAC/RANSAC.
3. Warp Yale pixels into a fixed 2× canonical page grid.
4. Use VT rectangles only to define the **outer envelope of a complete matched physical line**. Internal VT rectangle boundaries, word identities, word indices and topology are not supplied to the line splitter or outcome measurement.
5. Apply the fixed DINO adaptive ink mask: Gaussian adaptive threshold, block 31, C=12, followed by 2×2 morphological opening.
6. Apply the pre-existing DINO connected-component admission rule: retain 8-connected components with area >= 6 px.
7. Split the resulting single line mask into the boundary-stripped ZL glyph count using the same V02 low-ink dynamic-programming cuts and width regularisation. The optimiser sees pixels and glyph count only.
8. At each blinded adjacent `r→a` index, measure rightmost retained ink in the `r` slot to leftmost retained ink in the `a` slot, floored at zero.
9. Primary outcome is gap divided by robust 5–95% retained line-ink height. Raw registered-Yale pixels are mandatory sensitivity output.

No DINO embedding inference, Hugging Face Job or GPU is used.

## Alignment gates

Unchanged from V02:

- VT text is boundary-stripped and used only for matching ZL running-text lines to physical line envelopes;
- primary line similarity >= 0.45;
- both target glyph indices must map as exact consecutive `r` and `a` under monotone character alignment;
- similarity >= 0.70 sensitivity;
- exact boundary-stripped ZL=VT line sensitivity.

## Primary statistic and null

Unchanged from V02. Within each folio containing both classes, compute mean normalized gap(certain) − mean normalized gap(uncertain), weighted by `n_c*n_u/(n_c+n_u)`.

Permute certain/uncertain labels 9,999 times within folio, preserving class counts. Seed = `6037`.

Report actual effect, null mean, null SD, z, one-sided plus-one p, within-folio stratified AUC and a 5,000-replicate folio bootstrap 95% CI.

Primary is RESOLVED only if:

- p <= .05;
- effect >= 2 null SD above null mean;
- folio-bootstrap 95% CI is entirely >0;
- every finite leave-one-hand-out primary effect remains >0.

The three-way continuum is RESOLVED only if primary passes and uncertain-minus-joined has p <= .05 and z >=2.

## Mandatory measurement-validity gate

Before interpreting class differences, V03 must pass a physical-instrument validity check. Report:

- class-specific zero-gap rates;
- pooled zero-gap rate;
- retained line-ink-height / line-crop-height distribution.

The instrument is considered degenerate and its hypothesis result is not interpreted if either:

1. pooled zero-gap rate >= 0.75; or
2. median retained line-ink-height / crop-height >= 0.75.

These thresholds are deliberately coarse engineering gates, frozen before V03 outcome. They are not statistical evidence for the hypothesis.

## Mandatory bounds

- raw registered-Yale pixels;
- line similarity >=0.70;
- exact boundary-stripped lines;
- leave-one-folio-out;
- hand and leave-one-hand-out;
- class-specific retention;
- V01/V02 comparison only after V03 is complete and labelled cross-instrument sensitivity.

## Interpretation boundary

A positive V03 establishes only that ZL boundary confidence tracks physical ink separation under a boundary-blind Yale-line instrument. It does not establish wordhood, semantics, decipherment, or three discrete scribal categories.
