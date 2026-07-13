# Voynich DINOv3 to Nomic Atlas

This integration publishes the completed Voynich DINOv3 embeddings as interactive Nomic Atlas maps without replacing or re-embedding the source vectors.

## Deployed public maps

The production deployment was built from `Digitalgoldfish79/vdino3-crops` at source commit `b86d96600dc49c5298e07dcecb1ebdbd44970483`.

- Words: 37,886 normalized word crops across 225 folios.
  - Dataset: https://atlas.nomic.ai/data/edwardbozzard/voynich-dinov3-words-20260713-v1
  - Map: https://atlas.nomic.ai/data/edwardbozzard/voynich-dinov3-words-20260713-v1/map
- Proposals: deterministic stratified sample of 50,000 from 747,672 normalized high-confidence `cc` and `ccmerge` proposals.
  - Dataset: https://atlas.nomic.ai/data/edwardbozzard/voynich-dinov3-proposals-50k-20260713-v1
  - Map: https://atlas.nomic.ai/data/edwardbozzard/voynich-dinov3-proposals-50k-20260713-v1/map

Both datasets use 768-dimensional float32 DINOv3 vectors, L2-normalized before upload. Topic modelling and duplicate detection were disabled. Public API verification confirmed the expected row counts and at least one ready projection for each dataset.

## Frozen choices

- Primary map: all 37,886 normalized mapped-word crops.
- Secondary map: a deterministic 50,000-proposal diagnostic sample.
- Geometry: a copy of each vector is L2-normalized before upload because Atlas builds L2 neighbours; this preserves cosine ordering while leaving source files untouched.
- Topic modelling and duplicate detection: disabled by default.
- Visibility: private unless `--public` is explicitly supplied.
- Existing datasets: exact expected row counts are reused idempotently; unexpected non-empty datasets are not appended to unless `--allow-nonempty` is explicitly supplied.
- Existing ready projections are reused rather than duplicated.

Atlas neighbourhoods and 2-D islands are exploratory diagnostics, not glyph classes.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r tools/nomic_atlas/requirements.txt
```

## Prepare the word-level bundle

```bash
python tools/nomic_atlas/voynich_atlas.py prepare \
  --run-dir /path/to/runs/2026-07-13.full-corpus.v2.multiscale \
  --output-dir atlas_bundle_words \
  --run-id 2026-07-13.full-corpus.v2.multiscale \
  --level word \
  --normalize l2
```

The loader finds `outputs/word_embeddings.npz` or `word_embeddings.npz`. It imports any one-dimensional NPZ arrays whose row count matches the embedding matrix, and optionally merges a matching JSONL, JSON or CSV metadata file. Pass `--metadata` or `--embedding-key` when automatic discovery is ambiguous.

The preparation bundle contains:

- `atlas_embeddings.npy`
- `atlas_metadata.jsonl`
- `source_indices.npy`
- `atlas_manifest.json`, including source hashes, array inventory, dimensions and output hashes

Boolean metadata is encoded as integer flags because Nomic Atlas 3.9 does not accept Boolean Arrow columns.

## Publish

```bash
export NOMIC_API_KEY='...'
python tools/nomic_atlas/voynich_atlas.py publish \
  --bundle-dir atlas_bundle_words \
  --dataset voynich-dinov3-words \
  --index-name voynich-dinov3-word-map \
  --public
```

Publishing creates or loads an `AtlasDataset`, uploads metadata and matching DINOv3 vectors when the dataset is empty, reuses an exact existing row count, and creates an index only when no projection exists. It writes `atlas_publish_result.json` containing row counts, visibility, index status and resulting links.

## Proposal diagnostic map

```bash
python tools/nomic_atlas/voynich_atlas.py prepare \
  --run-dir /path/to/run \
  --output-dir atlas_bundle_proposals_50k \
  --run-id 2026-07-13.full-corpus.v2.multiscale \
  --level proposal
```

The generic proposal mode defaults to 50,000 deterministic rows. Use `--max-rows` to change the sample size. The production proposal bundle used a stronger proportional stratification over crop kind, folio and EVA alignment.

## Verification workflow

`.github/workflows/publish-nomic-atlas.yml` is now manual verification only. It checks the two public Nomic API records for exact row counts and ready projections; it performs no upload or deletion.

## Interpretation constraint

Any apparent Atlas cluster must be tested again in the original high-dimensional embedding space against folio, section, hand, crop scale, registration quality and proposal geometry. A 2-D separation alone has no evidential weight.
