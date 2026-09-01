# Final validation results ledger

## RETRACTED / DOWNGRADED FINDINGS

### 2026-09-01 — “Greek affinity survives removal of word-level segmentation” — **RETRACTED / NOT SUPPORTED**
The first locked segmentation-independent CPU test reversed direction. Fixed-patch morphology placed Voynich closer to the frozen Latin centroid than the frozen Greek centroid.

- `D(VMS,GREEK) = 1.0900816089`
- `D(VMS,LATIN) = 0.9538089917`
- Greek advantage `A = D_LATIN - D_GREEK = -0.1362726172`
- exact balanced-label null SD `= 0.1332979902`
- `Z = -1.0223950583`
- exact one-sided permutation `p = 0.8394716395`
- balanced null assignments enumerated: `12,869` (all C(16,8)=12,870 assignments except observed labeling)

Therefore the preregistered CPU gate fails. Under `FINAL_LOCKED_VALIDATION_PROTOCOL_20260901.md`, no DINO fixed-patch run is permitted from this primary result.

This does **not** erase the earlier empirical word-box results (CPU replication and DINO replication). It downgrades their mechanistic interpretation: the previously observed Greek affinity is currently dependent on, or at least not reproduced after removal of, the shared word/token segmentation stage.

## CURRENT RESULTS

### Primary fixed-patch CPU — frozen cohort — FAILED

Protocol: `2026-09-01.greek-affinity.final-fixedpatch.v1`

Headline result: see above.

Control diagnostic:
- leave-one-manuscript-out Greek/Latin accuracy: `0.6875` (11/16)

Gross-statistics confound null:
- `D(VMS,GREEK) = 1.8909766461`
- `D(VMS,LATIN) = 1.5752582683`
- Greek advantage `A = -0.3157183778`
- null SD `= 0.3164085225`
- `Z = -0.9978963543`
- exact one-sided `p = 0.8254079254`
- LOMO accuracy `= 0.625` (10/16)

Interpretation: gross image statistics do not independently generate the previous Greek direction; both headline fixed-patch morphology and gross-statistics null are Latinward in this representation.

Patch yields varied materially across control manuscripts (45–360 accepted patches after deterministic selection); this is a mandatory audit item and remains a limitation to bound.

### Stricter extraction sensitivity — CONFIRMS PRIMARY DIRECTION

Predeclared ink gate `[0.03, 0.30]`. Secondary only; cannot rescue the primary.

- `D(VMS,GREEK) = 1.1272332456`
- `D(VMS,LATIN) = 1.0001294558`
- Greek advantage `A = -0.1271037898`
- null SD `= 0.1266356958`
- `Z = -1.0037743763`
- exact one-sided `p = 0.8332556333`
- LOMO accuracy `= 0.625` (10/16)
- VMS accepted patches = `182`

The stricter gate reproduces the same Latinward direction and approximately the same standardized magnitude as the primary (`Z=-1.02`). This bounds the headline against the preregistered extraction-threshold sensitivity.

Gross-statistics sensitivity remains non-resolving and Latinward (`Z=-0.6172`).

### Davis-hand sensitivity — IN PROGRESS

The initial runner failed to match Yale canvas labels because the Yale manifest labels omit the leading `f` (e.g. `3r`, not `f3r`). This is a label-resolution implementation bug only; it did not affect the primary or gross-statistics result. A corrected mapping rerun is in progress. Mixed-hand `f115r` remains excluded by the frozen rule.

## PRIOR RESULTS RETAINED AS PRIOR EVIDENCE, NOT FINAL CONFIRMATION

- Original CPU primary: Greekward but formally unresolved at its preregistered gate (`Z=1.805`).
- Independent-sample word-box CPU replication: Greekward (`Z=2.702`).
- Frozen DINO word-box confirmation: Greekward (`Z≈2.760`).
- Page-specific artificial/cipher controls: ordinary Greek remained closer than the tested artificial/cipher loci.

These remain historically accurate experiment results. Their interpretation is now constrained by the new negative fixed-patch test.
