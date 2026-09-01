# Greek-ductus CPU primary result — 2026-09-01

Protocol: `2026-08-31.greek-ductus.v1`

This file records the preregistered primary CPU result and mandatory frozen sensitivities. It does not promote any sensitivity above the primary result.

## Primary result

- VMS distance to Greek centroid: **0.6083993955**
- VMS distance to Italian-Latin centroid: **1.3613346645**
- VMS distance to German centroid: **1.2006588472**
- Greek advantage A: **0.6725973604**
- permutation null mean: **0.0035150199**
- permutation null SD: **0.3706418862**
- preregistered Z: **1.8051989408**
- empirical one-sided permutation p: **0.0064996750**
- decision: **UNRESOLVED (Z < 2.0)**

The empirical p-value does not override the preregistered Z threshold.

## Control diagnostics

- LOO family accuracy: **0.3889** (7/18)
- Greek controls classified Greek: **6/6**
- Italian-Latin controls: 0/6 classified Italian-Latin (2 Greek, 4 German)
- German controls: 1/6 classified German (2 Greek, 3 Italian-Latin)

This shows that the fixed stroke representation separates the Greek family much more cleanly than it separates the two Latin-script control families. That limitation is material to interpretation.

## VMS page-block consistency

All **18/18** frozen VMS pages had positive Greek advantage relative to the average of Italian-Latin and German.

- median page Greek advantage: **0.5546440**
- IQR: **0.0763293**
- fraction positive: **1.000**

Thus the primary signal is distributed across the manuscript rather than driven by one sampled page, but this is descriptive and does not replace the manuscript-level primary statistic.

## Frozen sensitivities

| Sensitivity | Z | Interpretation |
|---|---:|---|
| Primary | **1.8052** | unresolved |
| Exclude preregistered low-information crops | **1.8527** | unresolved |
| Stricter adaptive threshold C=21 | **1.5278** | unresolved |
| Exclude Dresden | **1.9216** | unresolved |
| Exclude Leipzig | — | underidentified (Greek 5 / Italian 2 / German 1) |
| Exclude Ferrara `Cod.graec. 256` | **2.0647** | sensitivity only; cannot rescue primary |

Secure Byzantine/Greek-East pair (`Mscr.Dresd.Da.61`, `Mscr.Dresd.Da.47`) descriptive VMS distance: **0.7127615640**.

## Decision on DINO/GPU

The preregistration required CPU shape **and** DINOv3-B to reach Z >= 2.0 for a `survives` conclusion. The CPU primary result did not meet that gate. Therefore a DINO run cannot rescue this preregistered experiment.

Given limited GPU credits, **do not spend GPU on this primary experiment**. Any future DINO work should follow an independent replication with a separately frozen manuscript panel and should not be described as confirmation of this unresolved primary result.

## Audit trail

Primary HF job: `6a965c650718b0f6d890b42e`
Sensitivity HF job: `6a9661760718b0f6d890b4de`

The sensitivity runner is committed at `greek_ductus/run_cpu_sensitivities.py`. The primary extraction/feature/analysis files remain unchanged from their preregistered branch state.
