# DINOv3-B confirmation protocol — 2026-09-01

Status: FROZEN BEFORE DINO FEATURE EXTRACTION

Purpose: independent visual-representation confirmation of the CPU stroke-geometry replication. This cannot rescue or alter the preregistered CPU primary result.

## Inputs
- Controls: the 16 new Laurenziana manuscripts frozen in `replication1_manifest.json` (8 Greek, 8 Latin), all catalogued 1301–1400 and all on the same digitisation host.
- Voynich: the 18 complementary pages frozen in `replication1_manifest.json`.
- Image representation: black ink on white only, using the frozen `extract_cpu.py` segmentation and `norm_crop`; no parchment/background colour is retained.
- Control sampling: gather all crops from the six frozen pages per manuscript, then take exactly 90 evenly spaced crops in deterministic reading-order across the gathered list.
- Voynich sampling: gather crops from all 18 frozen pages, then take exactly 270 evenly spaced crops across the gathered list.

## Model
- `facebook/dinov3-vitb16-pretrain-lvd1689m`
- pinned revision: `5931719e67bbdb9737e363e781fb0c67687896bc`
- CLS embedding only, L2-normalized.
- No fine-tuning. No augmentation. No hyperparameter search. No ViT-L/7B run in this phase.

## Headline statistic
For each control manuscript, average its 90 unit CLS vectors and L2-normalize the resulting manuscript centroid. Average and normalize the eight Greek manuscript centroids and separately the eight Latin centroids. Average and normalize the 270 Voynich vectors.

Cosine distance is `1 - cosine_similarity`. Define

`A = d(Voynich, Latin) - d(Voynich, Greek)`.

Positive A favours Greek. Permute the 16 manuscript family labels 20,000 times, preserving the 8/8 split by permutation. Report

`Z = (A_obs - mean(A_null)) / sd(A_null)`

and a one-sided empirical permutation p-value. The DINO confirmation gate is `Z >= 2.0`.

## Audits
- Leave-one-manuscript-out Greek/Latin classification is reported descriptively.
- Voynich page-level Greek-vs-Latin distances are reported descriptively from the selected crops on each page.
- The complete input crop bank and float16 embedding bank are persisted to a private Hugging Face dataset repository together with hashes and metadata.
- Network/download failures may be retried, but sampling, representation, model, metric and decision rule may not change after this file is committed.
