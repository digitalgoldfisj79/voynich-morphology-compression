# Palatino 586 root comparison — first-pass closeout

Date: 2026-08-03

## Status

The first-pass root-cropping and controlled DINOv3 comparison completed. It does **not** support a claim that Palatino 586 roots are morphologically closer to Voynich roots than to the external herbal control.

This is an exploratory crop-construction result, not a source, exemplar, transmission, or common-workshop claim.

## Frozen components

- DINOv3 proposal runner: `p586_dinov3_extract.py`, commit `b73d4c6591136dc1937f1e6b3fa4022e26f26703`.
- Independent full-page locator and crop QA: `p586_qwen_locator.py`, commit `f51743699a0d13d859b5c9a79eafc147da19fc6c`.
- Page-level comparison runner: `p586_compare.py`, commit `23c80c675b28f193da9936e7473901855218d101`.

## Stage 1 — DINOv3 proposal scan

- 58 Palatino canvases: 22–79.
- 2,088 multiscale candidate windows.
- 116 retained exploratory proposals, two per page.
- Reference calibration: 128 usable Voynich roots versus 129 usable Voynich whole-plant crops.
- Group-held-out root-versus-plant AUC: 0.9970 ± 0.0031.
- Result SHA-256: `83ea018bf86e0ecb0808de81b166be57467c77472b4593cdf49945a870820e4d`.

This stage used Voynich root anchors and therefore was retained only as a proposal/disagreement arm, not as the final crop selector.

## Stage 2 — independent full-page locator

The full-page locator received no Voynich image, embedding, similarity, or DINO proposal.

- 58 pages processed.
- 136 root boxes returned across 55 pages.
- Pages 68 and 75 returned no root boxes.
- Crop QA collapsed to one class: 136 `PARTIAL_ROOT`, zero other labels.
- Result SHA-256: `91e46dafa95fca8e07ca0e6eb0cbdcf10ab91bd6ebf784a7c5dbbfba4a9f591a`.

The one-class crop-QA output is non-discriminating and was not treated as validation.

## Stage 3 — DINOv3 validity and controlled comparison

Primary data:

- 136 independently located Palatino crops from 55 pages.
- 128 Voynich root references.
- 129 Voynich whole-plant controls.
- 21 usable roots from BSB Cgm 728 as the external herbal reference arm.

### Crop validity

A Voynich root-versus-whole-plant classifier retained its grouped AUC of 0.9970, but the Palatino proposals were only weakly root-like under that classifier:

- median Palatino root probability: 0.4622;
- mean: 0.4673;
- fraction at or above 0.5: 0.1618 (22/136).

This indicates that the first-pass locator boxes are too broad or inconsistent to be accepted as a clean root corpus without further review.

### Naive unequal-pool result

With all 128 Voynich roots available against only 21 BSB roots:

- mean page-level top-1 similarity to Voynich: 0.7751;
- mean page-level top-1 similarity to BSB: 0.7597;
- mean difference: +0.01544;
- page-bootstrap 95% interval: +0.01136 to +0.01969;
- Voynich wins on 85.45% of pages.

This result is confounded by the much larger Voynich reference pool and is not accepted as evidence.

### Matched reference-pool result

Voynich was repeatedly subsampled to the same 21-root size as BSB, with 10,000 reference subsamples/permutations and Palatino page as the inference unit.

- mean top-3 Voynich-minus-BSB difference: −0.01110;
- reference-subsample interval: −0.02590 to +0.00072;
- permutation p: 0.9439;
- mean page-level Voynich win fraction: 0.3563.

Therefore the apparent Voynich advantage disappears after controlling reference-pool size. The controlled test provides no evidence that Palatino 586 is closer to Voynich than the external herbal control.

Result SHA-256: `05ccfc7641ce5d4b7307cb1647f049296cb10c5214db185131ed7680ccb515aa`.

## Interpretation

1. The manuscript block can be processed and a large provisional crop set can be generated.
2. The current automated boxes are not yet a clean root corpus: only 22 of 136 clear the descriptive DINO root-probability midpoint.
3. Raw nearest-neighbour browsing is strongly affected by reference-pool size.
4. Once that bias is controlled, Palatino 586 does not show Voynich-specific root enrichment in this first pass.
5. The high-value next step, if continued, is targeted manual or stronger model-assisted tightening of the 136 boxes, followed by a new frozen comparison against several matched herbal manuscripts—not threshold relaxation of this run.
