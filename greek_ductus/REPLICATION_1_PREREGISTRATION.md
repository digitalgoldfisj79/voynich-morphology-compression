# Greek-ductus independent replication 1 — preregistration

Protocol: `2026-09-01.greek-ductus.replication1`
Status: `FROZEN_BEFORE_REPLICATION_PIXEL_EXTRACTION`

## Motivation
The primary preregistered experiment returned a Greek-favouring but formally unresolved result (Z=1.805). This replication is not permitted to rescue or replace that primary result. It asks whether the same direction reproduces in an independent ordinary-script control panel under a tighter repository/date design.

## Control population
Repository/digitisation host: Biblioteca Medicea Laurenziana ContentDM only.
Catalogue date class: 1301-1400 only.
Eligible controls must be compound manuscript objects with at least 100 digitised pages and be explicitly associated by the Laurenziana catalogue record with either the Greek catalogue family (`codicum Graecorum`) or Latin catalogue family (`codicum Latinorum`, `codices Latini`, or equivalent Latin catalogue wording).

Every manuscript used in the primary control experiment is excluded.

For each script family independently, sort all eligible shelfmarks lexicographically and select 8 evenly spaced ranks using rounded `linspace(0, N-1, 8)`. No image, crop, feature, title subject, prior Voynich comparison, or palaeographic resemblance enters selection. The frozen selections are stored in `replication1_manifest.json`.

## Page sampling
Controls: six fixed page fractions `[0.13, 0.27, 0.41, 0.59, 0.73, 0.87]` per manuscript, rounded to the nearest available canvas index.

Voynich: use the same Yale IIIF manuscript but a complementary 18-page sample not intentionally reusing the primary page fractions: `[0.030, 0.085, 0.140, 0.195, 0.250, 0.305, 0.360, 0.415, 0.470, 0.525, 0.580, 0.635, 0.690, 0.745, 0.800, 0.855, 0.910, 0.965]`.

The same deterministic word-like crop detector, ink normalization, and `2026-08-31.stroke.v1` descriptor are reused unchanged.

## Inferential unit and statistic
The manuscript is the control inferential unit. For each manuscript, each descriptor dimension is summarized by median and IQR and concatenated. Robust scaling (median, MAD*1.4826) is fitted on the 16 controls only.

Let `D(V,G)` be the Euclidean distance / sqrt(p) from Voynich to the median Greek control centroid, and `D(V,L)` the corresponding Latin distance.

Replication Greek advantage:
`A = D(V,L) - D(V,G)`.
Positive means Voynich is closer to Greek.

Balanced null: permute the 16 manuscript labels while preserving exactly 8 Greek / 8 Latin, seed 409, 20,000 permutations. Report
`Z = (A_obs - mean(A_perm))/sd(A_perm)`.

Decision rule:
- `Z >= 2.0`: independent CPU replication passes.
- `1.0 <= Z < 2.0`: directionally supportive but unresolved.
- `Z < 1.0`: replication does not support the primary direction.

An empirical permutation p-value is reported but cannot override the Z rule.

## Mandatory diagnostics
- Greek and Latin observed distances.
- Per-page Voynich Greek advantage on all 18 complementary pages.
- Fraction of Voynich pages with positive Greek advantage.
- Leave-one-manuscript-out Greek-vs-Latin classification accuracy.
- Per-manuscript distances.
- Crop counts by manuscript/page.

## GPU gate
No DINO/GPU run is permitted merely because a sensitivity or p-value is favourable. DINOv3-B is released only if this independent CPU replication reaches `Z >= 2.0`. Even then, the original primary CPU result remains formally unresolved and must be reported as such; DINO is treated as an independent follow-up representation, not a rescue of the primary test.
