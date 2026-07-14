# Voynich DINOv3 Atlas v2

## Live maps

- [Enriched word map](https://atlas.nomic.ai/data/edwardbozzard/voynich-dinov3-words-enriched-20260713-v2/map): 37,886 normalized word crops.
- [Enriched proposal map](https://atlas.nomic.ai/data/edwardbozzard/voynich-dinov3-proposals-100k-enriched-20260713-v2/map): deterministic 100,000-item sample from 747,672 eligible high-confidence `cc` and `ccmerge` proposals.

The original v1 maps remain intact. Atlas v2 is versioned and non-destructive.

## Word-map annotations

Each word row carries folio, hand, Currier/register, section and frequency metadata, plus high-dimensional diagnostics:

- exact-label 20-neighbour purity;
- cross-folio exact-label purity;
- nearest-neighbour and cross-folio-neighbour cosine similarity;
- nearest same-EVA and nearest different-EVA occurrence;
- word outlier score;
- maximum cross-label alias count;
- two-mode assignment and silhouette for frequent words;
- review-queue category and priority.

Use the word map to investigate transcription aliases, visually multimodal EVA labels, hand-dependent allographs, Currier-visible production differences and crop-registration outliers. A visible 2-D island is a review lead, not proof of a glyph or lexical class.

## Proposal-map annotations

Each proposal row carries source word context, slot, aligned EVA character, hand, Currier and section, plus:

- 32-cluster assignment;
- distance to cluster medoid;
- cluster size;
- dominant EVA and EVA purity;
- dominant proposal kind and kind purity;
- local 10-neighbour EVA purity;
- proposal outlier score;
- normalized slot position;
- held-out-folio cluster-to-EVA prediction and correctness.

Use this map to audit segmentation, inspect cluster medoids and boundary cases, and identify recurrent subglyphs or crop-geometry classes. The clusters are diagnostic families, not a recovered Voynich alphabet.

## Review workbench

The generated workbench contains:

- 400 exact-positive pair candidates;
- 300 allographic-positive pair candidates;
- 300 hard-negative pair candidates;
- five medoids and five boundary cases for each of 32 proposal clusters, giving 320 proposal-review records;
- 350 available normalized crop thumbnails;
- ranked alias, multimodality and cluster-summary tables;
- a local HTML review interface with browser-local decisions and CSV export.

Candidate categories are machine-generated queues and must be human-adjudicated before use as a benchmark or training set.

## Reproducibility

- Source dataset: `Digitalgoldfish79/vdino3-crops`
- Source commit: `b86d96600dc49c5298e07dcecb1ebdbd44970483`
- Folio metadata commit: `85473f247f1221ca6e82f7103666b9d04de4f0f7`
- Embedding model: `facebook/dinov3-vitb16-pretrain-lvd1689m`
- Dimensions: 768
- Seed: `20260713`
- Full archive SHA-256: `8f4c529abf8aca2de7ec0a1c55cc5f05b3b5805e97de6614ed8f961c9ceabd15`

The full archive contains aligned metadata, float32 L2-normalized vectors, source indices, review outputs and manifests. See `DEPLOYMENT_V2.json` for project links, artifact identifiers and verification state.

## Atlas metadata-colouring limitation

Nomic SDK 3.9.0 automatically submits every non-internal metadata field as a colourable field when creating an embedding index. Both v2 projects store the enriched fields, but the current Atlas public API reports an empty `colorable_fields` array for their completed indices. The SDK provides no non-destructive method for patching that property on an existing map. The fields remain available in the project data and review exports; this limitation should be rechecked after a Nomic server or SDK update rather than worked around by deleting and recreating the projects.
