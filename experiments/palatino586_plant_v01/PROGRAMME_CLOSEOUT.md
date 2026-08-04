# Palatino 586 ↔ Voynich Plant-Morphology Programme — Closeout

Protocol: `P586-VMS-PLANT-0.1-20260803`

Status: **CLOSED**

## Conclusion

**Mixed or underpowered; no demonstrated Palatino–Voynich plant-morphology affinity.**

All three primary DINOv3 effects were directionally positive but small or unstable. Every hierarchical-bootstrap interval crossed zero, and no manuscript-label permutation test supported an affinity claim. The blinded vision adjudicator abstained on all 18 trials, so the blind stage is non-informative rather than positive or negative.

## Frozen corpora

- Palatino 586: 194 accepted whole plants; 194 valid masks; 194 valid above-ground crops; 712 reproductive proposals; 157 accepted and 1 partial reproductive structures.
- Voynich: 146 broad whole plants; 146 valid masks; 144 valid above-ground crops; 399 reproductive proposals; 122 accepted reproductive structures.
- Palatino final channel SHA-256: `e624603975b1202b299dfb364c68cddd391778e4d49d3a713588456313c6b8ce`
- Voynich final channel SHA-256: `dd5d92711b8bcb1a13f3921e9e04c616c5c07dea0b25f43cd6880b8ff23ce9f3`

## DINOv3 result

Model: `facebook/dinov3-vit7b16-pretrain-lvd1689m`

Revision: `b80367753773648a6793235ab9c65cdbb029506f`

Embeddings: 2,113; embedding errors: 0.

Result SHA-256: `534b5691450d745375fd2445ca085ca232a86116a5fe24c0ddbfba096b8c45f1`

Primary variants:

| Variant | Target minus control mean | 95% hierarchical-bootstrap interval | Manuscript-label p upper |
|---|---:|---:|---:|
| Whole masked broad | +0.0332904435 | [-0.0552293956, 0.1218968470] | 0.5 |
| Above context broad | +0.0340082446 | [-0.0534064831, 0.1206351028] | 0.5 |
| Flowers broad | +0.1346135099 | [-0.2896286432, 0.4498001095] | 0.4 |

## Blind stage

- Trials: 18
- Decisions: 18 abstentions
- Interpretation: non-informative; no post-result prompt or parser tuning
- Result SHA-256: `8ffc5fac83a5f7b1ba712634fc09bc3ecd8cc685871f9db3f155d79dad623616`

## Registration

The frozen Palatino corpus was registered idempotently in the generic Plant Mask Painter tables:

- `herbal_objects`: 194 plant objects
- `herbal_plants`: 194 plant rows
- unique `(slug, obj_index)` keys: 194
- ordinary crop paths: 194
- masked crop paths: 194

## Reports

- Closeout result SHA-256: `5985b7e3242d0f53cf656436e4a1a54c11966fb1b35be0e25e68d7b62eb34f15`
- Delivery metadata: `p586_plant_v01/results/human_bundle.json`

## Persistent delivery

### Compact core archive

Contains protocols, all amendments, reports, manifests, CSV ledgers and blind evidence sheets.

- Public path: `p586_plant_v01/bundles/P586_PLANT_MORPHOLOGY_HUMAN_READABLE_CORE.zip`
- Bytes: 4,848,220
- Entries: 89
- SHA-256: `1305582a80a2aef52d5cf54af65bfd29240c2e9d3b26f5bd999ca2f2bbbee364`
- ZIP integrity test: pass

### Complete evidence archive

Contains the core material plus every retained ordinary, masked, above-ground and reproductive crop.

- Bytes: 632,879,834
- Entries: 3,307
- SHA-256: `bfacee7506f8e201ea29bfb8e4440f862a4e6c976935f506e73cd12fdbfb9a3e`
- Delivery: 14 ordered binary parts
- Manifest SHA-256: `60fbe587050a2cfc19b62ab13464efd74f84b8666b7ec7df1a2d25ab55b3a4e5`
- Independent remote reconstruction: all 14 part hashes passed; complete archive hash passed; ZIP integrity test passed.
- Windows and Unix assembly scripts are stored beside the manifest.

## Closeout controls

- Plant Mask Painter temporary mask endpoint: HTTP 410.
- Supabase temporary corpus, registration, audit, upload, ticket, TUS, multipart-ticket and cleanup endpoints: HTTP 410.
- Temporary tunnel and HTTP-server jobs: cancelled.
- Temporary transfer pointer: deleted.
- Test upload objects: deleted.
- `bridge` bucket file-size limit restored to 104,857,600 bytes.
- Persistent delivery objects: one core ZIP, one manifest, fourteen complete-archive parts, three helper files and one delivery-metadata record.
- Active Hugging Face jobs at final closeout: zero.

No historical transmission, common-exemplar, or botanical-identity inference is established by this programme.
