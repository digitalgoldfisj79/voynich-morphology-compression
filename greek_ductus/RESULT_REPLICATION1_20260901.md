# Greek ductus replication 1 — result

Protocol: `2026-09-01.greek-ductus.replication1`

Status: **REPLICATION_PASS_Z_GE_2**

## Design

Independent control panel selected before pixel analysis: 8 Greek and 8 Latin manuscripts, all Biblioteca Medicea Laurenziana, all catalogued 1301–1400, all excluded from the primary experiment. Selection used evenly spaced catalogue ranks across the eligible Greek and Latin populations. Six fixed page fractions per manuscript were sampled. Voynich used 18 complementary fixed page fractions, distinct from the primary 18-page sample.

The feature representation is unchanged from the primary experiment: `2026-08-31.stroke.v1`. The manuscript is the inferential unit. The replication used a 20,000 balanced manuscript-label permutation null with seed 409.

## Headline result

- Voynich → Greek centroid distance: **1.0826168455**
- Voynich → Latin centroid distance: **3.3719396119**
- Greek advantage (`D_LATIN - D_GREEK`): **2.2893227664**
- Permutation null mean: **0.0009841045**
- Permutation null SD: **0.8470561591**
- Standardized effect: **Z = 2.7015194179**
- Empirical one-sided permutation p: **0.0041497925**
- Decision: **replication passes the predeclared Z >= 2 threshold**

## Internal diagnostics

Leave-one-manuscript-out script-family accuracy was **14/16 = 87.5%**. All 8/8 Greek controls classified as Greek. Six of eight Latin controls classified as Latin; two classified as Greek.

All **18/18 complementary Voynich pages** were closer to the Greek centroid than to the Latin centroid. Median page-level Greek advantage was **2.10735**, IQR **0.11190**, fraction positive **1.0**.

Control crops: 8,507. Voynich crops: 1,620.

## Execution incident and audit trail

The first replication execution (`HF job 6a966d2a21c5aa7c8364ab24`) failed at 2026-09-01 06:17 UTC because the Laurenziana ContentDM endpoint closed a connection mid-request (`RemoteDisconnected`). It failed before producing a result. The runner was then patched only for network fault tolerance: seven bounded retries with logged retry events. No manuscript selection, sampling fraction, feature extraction, distance metric, permutation rule, seed, or decision threshold changed.

The hardened execution (`HF job 6a96781d0718b0f6d890b851`) completed successfully. It encountered the same transient failure once while requesting the manifest for `Plut.1 sin.10`; the retry succeeded, demonstrating that the patch addressed the prior infrastructure failure without changing the measurement.

## Interpretation constraint

The original primary experiment remains formally unresolved at **Z = 1.805** and is not retroactively converted into a positive result. This independent same-host/same-century replication is a separate positive result. Together they justify proceeding to the preplanned independent DINOv3-B representation, but the DINO result must stand on its own and cannot alter either CPU result.
