# BSB Clm 5905 ↔ Voynich roots and flowers: corrected Qwen rerun v0.2

Protocol ID: `CLM5905-VMS-QWEN-RF-0.2-20260805`

Status at freeze: no v0.2 localization, embedding or similarity result exists.

## Purpose

Repeat the Clm 5905 comparison after invalidating v0.1. The new run must compare homologous visual components extracted and quality-controlled by the same model and rules on every manuscript.

## Frozen source panel

Target: BSB Clm 5905, 198 frozen whole-plant illustrations.

Reference: Voynich herbal whole-plant crops from the frozen P586 parent bundle.

Controls: `bnf_lat_6862`, `bnf_gr_2179`, `herb_18f0aa144a2b`, `herb_78e2bbc79062`, and `bsb1784`, using the same frozen parent bundle.

## Frozen localization model

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Deterministic decoding; temperature 0.
- Identical localization and crop-QA prompts for target, reference and controls.

## Component ontology

Root channel: a visible root system below the stem/root transition. Whole plants, basal leaf rosettes, stems and arbitrary red regions are forbidden.

Flower channel: only `flower`, `flower_head`, or `inflorescence`. Buds, fruits, berries, seed heads, leaves and stems are excluded from the strict channel. A broad sensitivity may add buds, but cannot alter the primary result.

## Two-stage Qwen gate

1. Qwen localizes candidate boxes on the whole-plant image.
2. Each candidate crop is independently re-presented to Qwen without the first answer. It must confirm the requested component, reject contamination or wrong classes, and assign confidence at least 0.75.

Geometry gates reject near-whole-image boxes, tiny boxes, roots whose centres are not in the lower portion of the plant, and reproductive boxes whose class is outside the frozen ontology.

## Deduplication

Exact SHA-256 duplicates and near-duplicate perceptual hashes are grouped before analysis. No duplicate source image may appear on both sides of a comparison. Corpus controls with duplicated reference content are excluded from the affected channel.

## Representation and analysis

After extraction freezes, embed the exact accepted ordinary crops with the frozen DINOv3 representation used in the parent programme. The displayed contact sheets must use the same crop bytes ranked by the model. Masked crops, if generated, are sensitivity-only and may not substitute for ordinary-crop rankings.

Primary estimand: manuscript-balanced target-minus-control mean of each query crop's best cosine similarity to the Voynich reference, separately for roots and strict flowers.

Required outputs include full counts, rejected-crop reasons, duplicate groups, leave-one-control-out effects, bootstrap intervals, ranked pairs, and byte-identical contact sheets.

## Interpretation

Individual nearest neighbours are descriptive only. A corpus-level affinity requires a positive primary effect, a 95% interval above zero, target rank 1, no duplicate leakage, and visual QA of the ranked contact sheet.

## Invalidated predecessor

All v0.1 effects, ranks, pairs and contact sheets remain withdrawn. They cannot be reused as evidence or as a prior for this rerun.
