# BSB Clm 5905 ↔ Voynich Root-and-Flower Programme v0.1 — Preregistration

Protocol ID: `CLM5905-VMS-RF-0.1-20260804`

Repository branch: `gpt56/clm5905-roots-flowers-v01-20260804`

Parent frozen comparison corpus: `P586-VMS-PLANT-0.1-20260803` and its published human-readable archive. No parent similarity outcome is used to choose Clm 5905 objects, thresholds, channels or exclusions.

## 1. Question and unit of inference

The primary question is whether the roots and flowers of München, Bayerische Staatsbibliothek, Clm 5905 (Vitus Auslasser, 1479) are unusually similar to Voynich herbal imagery relative to the fixed eligible manuscript controls inherited from the parent programme.

The manuscript is the experimental unit. Individual crops are nested measurements and do not supply independent manuscript degrees of freedom.

Two primary channels are frozen:

1. roots, bulbs and rhizomes;
2. flowers, flower heads and inflorescences.

Whole-plant morphology is a secondary sensitivity channel only. Fruit, seed heads and buds are reported separately and may enter a declared broad reproductive sensitivity analysis, but not the strict flower primary.

A channel-level positive result requires a positive balanced manuscript effect, a bootstrap interval excluding zero, manuscript-label permutation/rank support, and no contradiction in blinded visual adjudication. A strong dual-channel affinity requires both primary channels to satisfy these conditions. Attractive individual pairs are descriptive only.

## 2. Target source freeze

The target is exactly BSB Clm 5905, digital object `bsb00092488`, IIIF Presentation v2 manifest:

`https://api.digitale-sammlungen.de/iiif/presentation/v2/bsb00092488/manifest`

The target contains exactly 198 numbered herbal illustrations. The primary image set is frozen before extraction as follows:

- illustrations 1–83: folios 95r–177r in sequence;
- illustration 84: folio 179r;
- illustrations 84–198: folios 179r–293r in sequence;
- folio 178r is excluded because the published illustration sequence skips from 177 to 179.

Only these 198 recto canvases enter the target corpus. Canvas identifiers, labels, IIIF service URLs, requested image URLs, dimensions, byte lengths and SHA-256 hashes are persisted. Any mismatch between this rule and the live manifest stops the run.

## 3. Historical identifications

Hermann Fischer's 1925 numbered commentary is acquired and hashed. Its identification for each illustration is attached as metadata where parseable. The historical names and Fischer identifications cannot determine inclusion, exclusion, crop boundaries, model prompts, similarity thresholds or the primary statistical result. Question marks and alternative identifications are preserved rather than resolved silently.

Taxonomic identity is a post-primary stratification only. Same-name and related-taxon comparisons cannot replace the open-set manuscript-level test.

## 4. Primary extraction

Each frozen folio is processed without access to Voynich similarities. The page localiser returns every coherent plant and separate whole, above-ground and below-ground boxes. Because the source catalogue defines one numbered illustration per frozen folio, a page with zero or multiple plausible plants is retained in the ledger and receives explicit QA rather than being silently repaired.

For every retained plant:

- the whole-plant crop includes fixed 6% horizontal, 4% upper and 8% lower proportional padding;
- the strict root crop is the localiser's below-ground box with 12% padding;
- a broad root crop includes the strict root plus up to 8% of whole-plant height above the frozen root boundary;
- reproductive structures are proposed from the accepted whole plant, with at most five proposals in model order;
- strict flowers contain accepted `flower`, `flower_head` and `inflorescence` objects;
- broad flowers add accepted/partial `bud` objects;
- fruit and seed heads remain separate.

Allowed QA labels are `accept`, `partial`, `reject`, `uncertain`. Strict sets use `accept`; broad sets use `accept + partial`. `uncertain` objects are reported but excluded from inference. No result-dependent manual crop adjustment is permitted.

## 5. Masks and background control

Ordinary crops and deterministic white-background foreground crops are both preserved. The deterministic mask rule is frozen before embedding and applied symmetrically to target and any newly derived reference roots. Mask area, border contact and foreground-fraction diagnostics are recorded. Pathological masks remain in the ledger and are excluded only by frozen objective gates.

The ordinary-crop result is mandatory. Masked results are sensitivity analyses and cannot rescue a failed ordinary-crop primary.

## 6. Reference and control reuse

Voynich and manuscript-control manifests, crops and DINOv3 embeddings are reused only when their SHA-256 hashes match the parent archive. Existing frozen flower embeddings are reused directly.

The root channel is constructed deterministically from the parent frozen whole-plant crops and frozen root boundaries. No new root-boundary inference is run on Voynich or controls. These derived root crops are embedded under the same DINOv3 pipeline as Clm 5905 and frozen before target labels are opened.

Eligible controls are exactly the parent manuscripts with at least eight valid objects in the relevant channel. No manuscript is excluded because it resembles or fails to resemble Clm 5905 or Voynich.

## 7. Embeddings

The primary representation is the L2-normalised CLS token from:

`facebook/dinov3-vit7b16-pretrain-lvd1689m`

Resolved revision, preprocessing constants, package versions and image hashes are recorded. No fine-tuning, channel-specific calibration or result-dependent threshold selection is allowed.

## 8. Statistics

For every strict/broad channel and ordinary/masked variant, the programme reports:

- mean and median best-match cosine similarity to Voynich;
- deterministic balanced equal-size subsampling, 500 repeats;
- manuscript-level and hierarchical bootstrap intervals, 10,000 replicates;
- exact manuscript-label rank/permutation where feasible, otherwise 100,000 permutations;
- nearest-neighbour manuscript enrichment and top-k source shares;
- leave-one-control-manuscript-out sensitivity;
- target and reference sample sizes and all exclusion reasons.

The primary effect is Clm 5905's balanced manuscript score against Voynich minus the mean corresponding score of eligible control manuscripts against the same Voynich pool.

## 9. Blind adjudication

Up to six strongest Clm 5905–Voynich root pairs and six strongest flower pairs are matched to control pairs with similar DINO similarity. Sheets conceal manuscript identities and randomise whether the target is A or B. Allowed answers are `A`, `B`, `tie`, `abstain`. The adjudicator scores visible component morphology, not general artistic quality or presumed plant identity. Malformed answers count as abstentions.

## 10. Forum-highlighted and named cases

The forum comparisons involving Voynich f15v, f43v, `Serpentina`, `Naterwurtz` or *Polygonum bistorta* are sealed exploratory analyses. They are opened only after all primary manifests, embeddings, statistics and blind sheets have been frozen. They cannot affect the primary conclusion.

## 11. Provenance and stopping

Prompts, model IDs and revisions, source URLs, hashes, random seeds, software versions, costs and every exclusion reason are persisted. Failed phases are corrected under the same frozen rule or documented through a dated amendment; results are never silently overwritten.

At closeout all temporary upload endpoints are retired and all jobs launched by this protocol are verified complete, failed or cancelled. Unrelated pre-existing jobs are not touched.

Claims are labelled exact, machine-certified, empirical, heuristic or open.