# Infrastructure addendum — 2026-08-31

Protocol: `2026-08-31.greek-ductus.v1`

This addendum was written **before any Greek-vs-Voynich feature values were computed**.

## What failed
The preregistered primary ordinary-script panel used Bodleian IIIF for all three families to minimize digitization-stack confounding. Hugging Face Jobs could not establish a connection to `iiif.bodleian.ox.ac.uk`; repeated manifest requests timed out before image analysis.

No Bodleian page images, crops, CPU descriptors, DINO embeddings, family distances, or Voynich comparisons were produced. Therefore there is no outcome information on which to tune manuscript replacement.

Recorded jobs:
- `6a9600f921c5aa7c8364a480`: environment failure (`cv2` unavailable), no data inspected.
- `6a96013c0718b0f6d890a706`: CPU audit cancelled during serial Bodleian connection waits, no scientific result.
- `6a9602770718b0f6d890a72d`: bounded parallel reachability audit; Bodleian connections timed out. Cancelled.

## Replacement rule
The original Bodleian panel remains frozen as the preferred replication panel if the host becomes reachable later.

For the operational primary run, manuscripts are replaced solely by:
1. script family / catalogue language or production region;
2. date overlap with 1300–1450 (a range may extend slightly beyond 1450 if it overlaps the window, as allowed by the original inclusion rule);
3. stable IIIF manifest on a host demonstrated reachable from the HF execution environment;
4. sufficient page count for deterministic fixed-fraction sampling;
5. no prior selection for resemblance to Voynich.

Reachability-only test job `6a96036821c5aa7c8364a49b` established HTTP 200 manifest access for Leipzig, Dresden, Florence ContentDM, BSB and Yale. Trinity timed out and is excluded operationally for the same infrastructure reason.

## Family interpretation
`GREEK` tests Greek-minuscule **script practice**, not production in Constantinople. Secure Byzantine/Greek-East provenance is retained as metadata and analysed as a sensitivity subset. A Greek manuscript copied in Italy is not silently reclassified as Byzantine production.

`ITALIAN_LATIN` requires catalogue evidence of Italian production/origin, not merely current storage in Italy.

`GERMAN` requires catalogue evidence of German/Central-German production/origin, not merely current storage in Germany.

## Digitization confound mitigation
Because the operational panel necessarily spans repositories, the primary CPU representation uses normalized binary ink/skeleton geometry. DINO, if released by the CPU stop rule, is run on normalized black-on-white crops rather than raw parchment scans. Repository/host is retained as a nuisance label; repository prediction and leave-host-out sensitivity are mandatory diagnostics.

No host/repository-specific result may be interpreted as palaeographic affinity.
