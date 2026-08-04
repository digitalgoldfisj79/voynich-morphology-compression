# Invalidation notice — Clm 5905 ↔ Voynich roots and flowers v0.1

Date: 2026-08-04
Protocol: `CLM5905-VMS-RF-0.1-20260804`
Previous result SHA-256: `d5fc3d7b913dd218946e7e6f9b8f5ad9606ed9d949f0528ca62e7945a1710ea1`

## Status

**INVALID — do not interpret either the reported negative corpus-level effect or the listed nearest-neighbour pairs.**

A human-readable contact-sheet audit exposed failures that were not visible in the aggregate statistics.

## Fatal defects

1. **The contact sheet did not display the representation used for ranking.** The primary rankings were calculated from `ordinary_path` embeddings, but the generated contact sheet displayed `masked_path` images. It therefore did not faithfully visualize the reported comparisons.

2. **The Voynich root channel was not reliably root-localized.** Parent roots were derived primarily from a red-colour mask over the full plant crop. Several returned bounding boxes span the entire source crop (`x=0`, `y=0`, full width/height), so Clm root crops were compared against complete or substantially complete Voynich plants rather than homologous root components.

3. **The flower channel was not class-balanced.** Parent strict flowers were restricted to `flower`, `flower_head`, and `inflorescence`, while Clm target rows admitted every accepted reproductive proposal without an equivalent class filter. This included buds, fruits, seed heads, and mislocalized leaves/stems. The reported strict Clm count (`658`) is inconsistent with the frozen strict flower count (`541`) and demonstrates the mismatch.

4. **Nearest-neighbour concentration and blind rationales indicate localization failure.** Most leading flower matches collapsed onto two Voynich crops, while blind rationales repeatedly described leaves, stems, and branching rather than floral morphology.

5. **Control scores contain exact or near-exact duplicate leakage.** A control corpus reported mean and median best similarity approximately `1.0`, invalidating the manuscript-control baseline unless deduplicated at image and source-object level.

## Consequence

The previous classification `no_corpus_level_affinity` is withdrawn. The valid conclusion is only that **v0.1 failed as a measurement pipeline**. A corrected run requires homologous component segmentation on both sides, identical class gates, ordinary/masked representation parity, source-object deduplication, and human QA before embedding or statistical testing.
