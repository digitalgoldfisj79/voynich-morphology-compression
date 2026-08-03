# Palatino 586 ↔ Voynich Plant-Morphology Programme v0.1 — Preregistration

Protocol ID: `P586-VMS-PLANT-0.1-20260803`

Repository branch: `gpt56/p586-plant-v01-20260803`

Parent frozen root result: `P586-VMS-ROOT-0.1-20260803`; result SHA-256 `588e0aa0a759f84fd8468c043f3fb3eb17b173f0c2fc4af3d186a351148eb206`.

## 1. Question and unit of inference

The primary question is whether Firenze, Biblioteca Nazionale Centrale, Palatino 586 is unusually similar to Voynich herbal imagery relative to a fixed panel of manuscript controls. The experimental unit is the manuscript. Crop-level matches are descriptive measurements nested within manuscripts and are never treated as independent manuscript replicates.

The three primary channels are:

1. whole plants;
2. root-excluded above-ground morphology;
3. flowers, inflorescences and reproductive structures.

A positive Palatino affinity requires a positive manuscript-level effect in at least two channels, at least one of which is above-ground or reproductive morphology, a bootstrap interval excluding zero, manuscript-label permutation support, and no contradiction from blinded visual adjudication.

## 2. Identity and source freeze

The target is exactly the Supabase manuscript record `bncf_palatino_586`: Firenze, Biblioteca Nazionale Centrale, Palatino 586 (Detti di filosofi; Aforismi di medicina; Erbario), manifest `https://iiif.archive.org/iiif/bncf-pal.-586-images/manifest.json`.

The target page set is exactly the 66 page records in the corrected frozen root checkpoint. The new run records their canvas indices, labels, source URLs, image dimensions and SHA-256 hashes. It must contain exactly 66 distinct canvases. Existing root-context crops are localisation hints and QA cross-checks only; they cannot create, remove or duplicate a whole-plant observation.

No `herb_*` identifier is an alias for Palatino 586.

## 3. Whole-plant extraction and QA

The detector is `Qwen/Qwen2.5-VL-7B-Instruct`, deterministic decoding, fixed prompt and fixed pixel limits. Every botanical illustration on every target page is proposed independently of roots. Multiple plants per page are permitted. Plant proposals are deduplicated at page level when intersection-over-union exceeds 0.65, retaining the higher-confidence proposal.

Whole boxes include fixed proportional padding: 6% horizontally, 4% above and 8% below. Proposals are visually adjudicated before any DINOv3 embedding or Voynich similarity is computed.

Allowed QA labels are `accept`, `partial`, `reject`, `uncertain`:

- `accept`: a complete or effectively complete coherent plant, with no material truncation;
- `partial`: a morphologically useful plant with material truncation or obstruction;
- `reject`: not a plant, duplicate, decorative fragment, text, figure or unusable crop;
- `uncertain`: genuine ambiguity that cannot be resolved without outcome information.

The strict corpus is `accept`. The broad sensitivity corpus is `accept + partial`. `uncertain` is reported but excluded from both inferential sets. A deterministic cap of 20 accepted/broad plants per control manuscript is applied in source-page and proposal order; Palatino and Voynich are not capped during extraction. The whole-plant manifest and its hash are frozen before masking, channel extraction or embeddings.

## 4. Fixed control panel

Main manuscript controls are fixed before target comparisons:

- `bnf_lat_6862` — BnF Latin 6862;
- `herb_8510642bf2ce` — Oxford, Bodleian MS Bodl. 130;
- `herb_c51d653739d2` — BnF Grec 27;
- `herb_0b5263630edb` — Florence, Laurenziana Plut. 18 sin. 7;
- `herb_18f0aa144a2b` — Paris ENSBA Ms. Mas 116;
- `herb_eaec4fe75d89` — Munich BSB Cgm 9095;
- `herb_78e2bbc79062` — BnF Latin 9333;
- `herb_d91d01bd5276` — St. Gallen Cod. Sang. 754;
- `bsb1784` — Munich BSB Cod. 1784.

For controls sourced from `cat_herbal_folios`, ten pages are selected by fixed equally spaced quantiles between the 5th and 95th percentiles of each manuscript's registered folio sequence. The page selection is frozen before visual detection. Existing BSB whole-plant crops are sampled deterministically in object order and reprocessed under the same mask and DINO pipeline.

A manuscript enters a channel's main inferential panel only if it has at least eight valid objects in that channel's broad set. No manuscript is excluded based on similarity to Voynich or Palatino.

Known-answer checks use the exact duplicate BnF Latin 6977 registry pair (`herb_7ce7efc90e6d`, `herb_fafef9a26da5`) and the two independent Ms. Mas 116 manifest records (`herb_18f0aa144a2b`, `herb_205bfb89efbc`). Duplicate/near-duplicate retrieval must exceed manuscript-label null performance.

## 5. Masks

The fixed mask model is `facebook/sam2.1-hiera-large`, prompted only by the preregistered plant box and without access to similarity outcomes. Its model revision, processor configuration and software versions are recorded. The highest predicted-IoU mask is retained, then subjected to fixed connected-component cleanup. Ordinary and white-background masked crops are preserved with hashes. Mask area fraction and border-touch diagnostics are recorded; failures remain in the ledger and are excluded only by preregistered gates (`0.01 <= area fraction <= 0.90`, no pathological full-border mask).

The same masking policy is applied to target and controls.

## 6. Above-ground channel

Before embeddings, Qwen visually assigns a root boundary on each accepted/broad whole plant. `root_boundary_y_1000` is the first horizontal row below which the coherent root, bulb or rhizome begins. If no root is visible, the boundary is 1000 and the whole masked plant is retained.

- strict above-ground crop: rows above the frozen boundary;
- context above-ground crop: strict crop plus 5% of whole-plant height below the boundary, bounded by the crop.

Boundary decisions cannot be changed after any DINOv3 similarity is observed.

## 7. Reproductive structures

A dedicated visual pass runs on accepted masked whole plants, not folio pages. At most five proposals per plant are retained in model order. Classes are kept distinct: `flower`, `flower_head`, `inflorescence`, `bud`, `fruit`, `seed_head`.

Every proposed sub-object receives a separate visual QA label using the same four-label vocabulary. Strict reproductive sets contain accepted `flower`, `flower_head` and `inflorescence` crops. Broad flower sets add accepted/partial `bud` crops. `fruit` and `seed_head` remain a separate channel and may be pooled only in a declared broad reproductive sensitivity result. Crop padding is fixed at 12% in each direction.

## 8. DINOv3 embeddings

The legacy 3,072-dimensional `herbal_objects.embedding` vectors are not assumed to be DINOv3. Repository provenance identifies them as `google/gemini-embedding-2`; they are not used as DINO features.

All images in this experiment are re-embedded with one frozen pipeline using `facebook/dinov3-vit7b16-pretrain-lvd1689m`. The exact resolved model revision and preprocessing constants are written to the run report. Preprocessing is implemented explicitly from the resolved processor configuration and hashed; the experiment does not depend on an unrecorded processor file. The L2-normalised CLS token is the primary representation. No channel-specific fine-tuning or threshold selection is permitted.

Separate embeddings are produced for ordinary whole plants, masked whole plants, strict above-ground, context above-ground, strict flowers, broad flowers, and fruit/seed-head structures where sample size permits.

## 9. Statistics

For every channel and strict/broad variant:

- mean and median best-match cosine similarity to Voynich;
- deterministic balanced equal-size subsampling, repeated 500 times;
- manuscript-level and hierarchical bootstrap intervals (10,000 replicates);
- manuscript-label permutation/rank test (exact where feasible, otherwise 100,000 permutations);
- nearest-neighbour manuscript enrichment;
- top-1, top-5 and top-10 source shares;
- leave-one-control-manuscript-out sensitivity;
- masked versus unmasked sensitivity.

The primary effect is Palatino's balanced manuscript score minus the mean score of eligible control manuscripts against the same Voynich reference pool. Individual crop pairs do not supply independent degrees of freedom.

## 10. Blind visual adjudication

For each channel, up to six strongest Palatino–Voynich candidate pairs are matched to controls with similar DINO similarity. Blinded sheets randomise whether Palatino is A or B, conceal manuscript identities and permit `A`, `B`, `tie`, or `abstain`. The adjudicator scores visible morphological correspondence, not beauty or general artistic quality. Malformed responses are recorded as abstentions.

## 11. Sealing, provenance and stopping rules

Extraction and QA manifests are frozen and hashed before embeddings. Embeddings are frozen before statistical code opens target labels. Prompts, model IDs, revisions, preprocessing, package versions, source URLs, image hashes, crop hashes, mask hashes, random seeds and every exclusion reason are persisted.

No post-result threshold tuning is allowed. If a phase fails, the failure is corrected by rerunning the same frozen rule or by preregistering a new version; the current result is not silently overwritten.

At closeout all temporary upload endpoints are disabled and all queued/running Hugging Face jobs are cancelled or verified absent.

Claims are labelled as exact, machine-certified, empirical, heuristic or open.
