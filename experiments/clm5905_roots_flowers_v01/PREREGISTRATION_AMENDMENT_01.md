# Preregistration Amendment 01 — implementation audit corrections

Protocol: `CLM5905-VMS-RF-0.1-20260804`

Date: 2026-08-04

This amendment is written after the first analysis implementation completed but before any result is accepted as a protocol result. The frozen 198-folio Clm 5905 extraction, its object statuses, boxes, source images and hashes remain unchanged.

## Defect A — target reproductive class filter

The v0.1 analysis loader admitted every accepted/partial reproductive object on the Clm 5905 side, while the parent side correctly admitted only the frozen flower classes. Consequently fruit and seed-head objects could enter the nominal flower comparison. The corrected implementation applies the preregistered class gates symmetrically:

- strict flowers: `flower`, `flower_head`, `inflorescence`, status `accept`;
- broad flowers: the strict classes plus `bud`, status `accept` or `partial`;
- `fruit` and `seed_head` never enter either flower analysis.

This is an objective schema correction. No crop is added, removed or re-labelled in response to similarity.

## Defect B — unsigned arithmetic in parent red-mask detection

The v0.1 parent-root derivation compared uint8 channels using expressions equivalent to `red >= green + 50`. Addition overflowed at 255 for bright parchment pixels, causing many derived root boxes to expand to nearly the entire parent whole-plant crop. The corrected implementation converts all colour channels to signed 16- or 32-bit integers before thresholding. It also records root bounding-box fractions and rejects objectively pathological boxes that occupy more than 95% of both source width and height unless the frozen mask itself contains red foreground at the border.

This is an objective numerical correction to the frozen colour rule. Clm 5905 root crops are unaffected.

## Defect C — representation symmetry audit

Before the corrected result is accepted, the implementation must verify that target and reference/control objects in each nominal ordinary or masked channel use the same representation family. Exact duplicate image hashes or embedding vectors across different manuscript identifiers are reported and excluded only when they demonstrate corpus leakage or duplicate registration; ordinary iconographic similarity is not grounds for exclusion.

## Status of v0.1 output

The v0.1 output with result SHA-256 `58353301beffe8e8ae6c8361fdebe076aa075df0f0654139f0665cdcf057b2c8` is retained as a failed implementation diagnostic. Its classification must not be cited as the programme conclusion.
