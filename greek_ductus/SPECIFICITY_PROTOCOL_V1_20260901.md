# Greek Ductus Specificity Test v1

Date: 2026-09-01
Status: FROZEN BEFORE ANY SPECIAL-COMPARATOR SIMILARITY RESULT

## Question
Does the previously observed Voynich→Greek-minuscule affinity remain when strong nonstandard-script competitors are admitted?

## Ordinary-script reference families
Reuse the frozen same-host Laurenziana replication panel: 8 Greek and 8 Latin manuscripts, all catalogued 1301–1400. These retain manuscript-level inference.

## Frozen special comparators
1. FONTANA_BELLICORUM: BSB Cod.icon.242, Venice 1420–1430, artificial Fontana cipher; IIIF manifest fixed in `specificity_manifest_v1.json`.
2. FONTANA_SECRETUM: BnF NAL 635, c.1430, artificial Fontana cipher; Gallica IIIF.
3. FOXTON_MIXED: Trinity R.15.21, 1408, Latin manuscript containing Foxton's artificial cipher; whole-page sampling is acknowledged as diluted by ordinary Latin.
4. BYZ_CIPHER_MIXED: BSB Cod.graec.201, dated 1267; final six canvases only, chosen before image inspection because the manuscript's concluding cryptographic material is documented. This is a Byzantine-cryptographic mixed-page comparator, not a pure cipher corpus.
5. LJS51_SECRET: Penn LJS 51, Syria? 15th c., collection of c.150 secret alphabets plus Arabic transcriptions. Fixed evenly spaced OPenn web images; acknowledged as mixed secret-script/Arabic.
6. EASTERN_ORDINARY_LJS49 and EASTERN_ORDINARY_LJS189: ordinary Persian/Arabic manuscript controls from OPenn, included so any LJS51 affinity cannot simply be labelled 'eastern/Arabic'.

## Sampling
- Ordinary Greek/Latin: reuse replication-1 six page fractions and deterministic 90-crop manuscript downsample.
- Voynich: reuse the 18 complementary replication-1 page fractions; deterministic 270-crop downsample.
- Each special comparator: 6 fixed pages/images; up to 90 normalized word-like crops per page, then deterministic 180-crop object downsample (or all if fewer).
- No page is replaced because it looks unhelpful.

## Representation 1: CPU stroke/geometry
The same black-on-white normalization is used for every object. Descriptor is fixed before execution: tight-box aspect and density, connected-component count, skeleton length/area, endpoint and junction density, Euler-hole count, 12-bin gradient-orientation histogram, 16-bin horizontal projection, 8-bin vertical projection, and log-Hu moments. Features are robust-scaled using all non-Voynich controls only. Object vectors are coordinate-wise medians of crop descriptors.

## Primary specificity readout
Distances are Euclidean in robust-scaled descriptor space.
- `d_G` = distance Voynich→mean ordinary-Greek object centroid.
- `d_L` = distance Voynich→mean ordinary-Latin centroid.
- `d_special[k]` = distance to each frozen special comparator object.
- `specificity_margin = min_k(d_special[k]) - d_G`.
Positive margin means ordinary Greek remains closer than every special comparator. Negative means at least one special comparator absorbs the prior Greek signal.

This margin is descriptive because rare comparators are not independent manuscript families of adequate n. No crop-level p-value may be used for them.

## Manuscript-level sanity checks
- Recompute Greek-vs-Latin leave-one-manuscript-out classification on the 16 ordinary controls.
- Recompute the frozen Greek-vs-Latin 20,000-label permutation Z for Voynich. This is a reproduction audit, not a new discovery test.

## Representation 2: DINOv3-B
Run only after CPU extraction/QC completes. Pinned model `facebook/dinov3-vitb16-pretrain-lvd1689m`, revision `5931719e67bbdb9737e363e781fb0c67687896bc`; normalized CLS embeddings; no tuning or training. Use the exact same frozen crops. Repeat the specificity-margin calculation in cosine distance.

## Decision language
- If Greek remains closest in both CPU and DINO: 'Greek affinity survives the strongest available artificial-script exemplars.'
- If a special comparator is closer in both: 'The earlier Greek result is better explained by a specific artificial-script comparator.'
- If representations disagree: 'specificity unresolved'.

No result localizes the Voynich manuscript geographically and no result identifies its language or cipher system.
