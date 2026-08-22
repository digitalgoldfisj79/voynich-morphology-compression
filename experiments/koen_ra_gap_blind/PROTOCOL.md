# KOEN_RA_GAP_V02_BOUNDARYBLIND_20260822 — frozen protocol

Frozen before any V02 physical-gap outcome is calculated.

## Why V02 exists

V01 used Voynichese.com/VT word rectangles as character-localisation windows. VT is mainly Takahashi-derived, so its internal word segmentation is not an independent observer of ZL spacing. The V01 numerical measurements remain an audit result, but the claim that the VT-cross-word sensitivity constituted an independent-transcriber control is retracted.

V02 removes **all VT/Takahashi internal word-boundary x coordinates and topology from the physical outcome path**. VT is permitted only as a page/line locator: its word rectangles may be grouped into a complete physical line envelope, and its boundary-stripped character stream may be used to match a ZL running-text line to that physical line. Once the line is identified, all internal VT word boxes, word identities, word indices, separators and topology are discarded.

## Question

For a fixed EVA `r→a` transition, do ZL uncertain spaces occupy a physically intermediate separation regime between joined `ra` and certain `r.a` when the gap is measured from a **single continuous Yale line image whose segmentation has never been shown a word boundary**?

Primary falsification target: within folio, is the physical `r→a` gap at ZL certain boundaries larger than at ZL uncertain boundaries?

## Fixed classes

Zandbergen–Landini IVTFF v3b running-text (`P`) loci only:

1. `joined`: literal plain-EVA `ra` inside a token;
2. `uncertain`: plain token ending `r`, ZL comma, next plain token starting `a`;
3. `certain`: plain token ending `r`, ZL period, next plain token starting `a`.

Drawing intrusions `<->`, alternative readings, unknown/high-ASCII/markup-defined target events and non-running loci cannot define an event. No other glyph pair may be substituted after target opening.

## Blinding / leakage firewall

The experiment writes two files before image measurement:

- `targets_blinded.json`: event id, folio, locus and boundary-stripped ZL character indices only; **no class label**;
- `labels_sealed.json`: event id → `joined|uncertain|certain`, plus a SHA-256 digest.

The image-measurement stage receives `targets_blinded.json` but not `labels_sealed.json`. It writes `measurements_blind.csv`. Only after that file is complete are labels joined for inference.

ZL punctuation/space marks are deleted before line-image segmentation. Thus `ra`, `r,a` and `r.a` all present identically to the physical instrument as adjacent character indices `r→a`.

## Yale physical instrument

1. Fetch the current Yale Beinecke IIIF manifest and a fixed-width Yale derivative for each required canvas.
2. Register the legacy page raster to Yale with the existing DINO CPU registration method: SIFT + CLAHE + USAC_MAGSAC/RANSAC.
3. Warp the Yale derivative into a fixed 2× canonical page grid. The resulting pixels come from Yale; the legacy raster supplies only the geometric coordinate frame.
4. Use VT rectangles only to identify the **outer envelope of the complete matched text line**. Internal rectangle x boundaries are not exposed to the character splitter or gap measurement.
5. Adaptive ink mask is fixed to the DINO baseline: Gaussian adaptive threshold, block 31, C=12, followed by 2×2 morphological opening.
6. Split the single continuous line raster into exactly the number of boundary-stripped ZL glyph positions using a low-ink dynamic-programming cut sequence with fixed width regularisation. The optimiser sees pixels and glyph count only; it never sees ZL spaces, ZL comma/period class, VT word boundaries or VT topology.
7. At each blinded `r→a` character boundary, measure the distance from the rightmost detected ink in the `r` slot to the leftmost detected ink in the adjacent `a` slot, floored at zero.
8. Primary outcome divides the canonical registered-Yale gap by a robust physical line-ink height. Raw registered-Yale pixels are mandatory sensitivity output.

No DINO embedding inference, Hugging Face Job or GPU is used.

## Line locator and alignment gate

VT/Takahashi text may be used only after deleting all separators, solely to match ZL running-text lines to physical line envelopes. It is not used to place character cuts.

- Primary line-string similarity threshold: >= 0.45, inherited from V01.
- Both target indices must be exact `r` and `a` matches in the boundary-stripped VT line under monotone SequenceMatcher alignment.
- Fixed stricter similarity >= 0.70 sensitivity.
- Fixed exact-line sensitivity: boundary-stripped ZL line string must equal the boundary-stripped VT line string exactly.

## Primary statistic and null

VT topology is not available to inference in V02. Within each folio containing both classes, compute mean normalized gap(certain) − mean normalized gap(uncertain), weighted by `n_c*n_u/(n_c+n_u)`.

Permute certain/uncertain labels 9,999 times **within folio**, preserving exact class counts. Seed = `6037`.

Report actual effect, null mean, null SD, z, one-sided plus-one p, within-folio stratified AUC and a 5,000-replicate folio bootstrap 95% CI.

Primary physical distinction is RESOLVED only if all hold:

- permutation p <= .05;
- observed effect is >= 2 null SD above null mean;
- folio-bootstrap 95% CI is entirely > 0;
- every finite leave-one-Davis-hand-out primary effect remains > 0.

## Secondary continuum claim

The three-way continuum is RESOLVED only if the primary passes **and** the within-folio uncertain-minus-joined comparison has p <= .05 and z >= 2.

Group distributions and overlap are reported regardless.

## Mandatory bounds

- raw registered-Yale pixels;
- line similarity >= 0.70;
- exact boundary-stripped ZL=VT lines only;
- leave-one-folio-out primary effects;
- Davis/ZL hand-specific and leave-one-hand-out effects;
- class-specific measurement retention rates;
- comparison of V02 with the V01 direction and magnitude, labelled as cross-instrument sensitivity rather than independent replication.

## Interpretation boundary

A positive result establishes only that ZL boundary confidence tracks a physical ink-separation variable under a boundary-blind Yale-line measurement. It does not establish linguistic wordhood, semantics, decipherment, or three naturally discrete scribal categories.
