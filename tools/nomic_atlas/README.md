# Voynich DINOv3 to Nomic Atlas

This integration publishes the completed Voynich DINOv3 embeddings as an interactive Nomic Atlas map without replacing or re-embedding the source vectors.

## Frozen choices

- Primary map: all mapped words from `2026-07-13.full-corpus.v2.multiscale`.
- Secondary map: a deterministic 50,000-proposal sample by default.
- Geometry: a copy of each vector is L2-normalised before upload because Atlas builds L2 neighbours; this preserves cosine ordering while leaving source files untouched.
- Topic modelling and duplicate detection: disabled by default.
- Visibility: private unless `--public` is explicitly supplied.
- Existing datasets: never deleted or reset; appending to a non-empty dataset requires `--allow-nonempty`.

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

## Publish

```bash
export NOMIC_API_KEY='...'
python tools/nomic_atlas/voynich_atlas.py publish \
  --bundle-dir atlas_bundle_words \
  --dataset voynich-dinov3-words \
  --index-name voynich-dinov3-word-map
```

Publishing creates or loads an `AtlasDataset`, uploads metadata and the matching DINOv3 vectors together, then calls `create_index(modality="embedding")`. It writes `atlas_publish_result.json` containing the resulting links.

## Proposal diagnostic map

```bash
python tools/nomic_atlas/voynich_atlas.py prepare \
  --run-dir /path/to/run \
  --output-dir atlas_bundle_proposals_50k \
  --run-id 2026-07-13.full-corpus.v2.multiscale \
  --level proposal
```

The proposal mode defaults to 50,000 deterministic rows. Use `--max-rows` to change the sample size.

## Interpretation constraint

Any apparent Atlas cluster must be tested again in the original high-dimensional embedding space against folio, section, hand, crop scale, registration quality and proposal geometry. A 2-D separation alone has no evidential weight.
