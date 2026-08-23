#!/usr/bin/env python3
# Plumbing-only schema probe; retained for audit. Frozen V05 source/protocol live in compressed payloads.
# This file does not inspect Koen ra/r,a/r.a outcomes.
import json, os, subprocess, sys

IDENTS = [
 'edwardbozzard/voynich-dinov3-proposals-100k-enriched-20260713-v2',
 'edwardbozzard/voynich-dinov3-words-enriched-20260713-v2',
 'edwardbozzard/voynich-dinov3-words-v3-overlay',
]
print(json.dumps({'nomic_secret_present': bool(os.getenv('NOMIC_API_KEY')), 'hf_secret_present': bool(os.getenv('HF_TOKEN'))}), flush=True)
key = os.getenv('NOMIC_API_KEY','')
if not key:
    raise SystemExit('NOMIC_API_KEY secret not present')
subprocess.run([sys.executable, '-m', 'nomic.cli', 'login', key], check=True, stdout=subprocess.DEVNULL)
from nomic import AtlasDataset

for ident in IDENTS:
    print('\n===', ident, '===', flush=True)
    ds = AtlasDataset(identifier=ident)
    print('TOTAL', int(ds.total_datums), flush=True)
    print('SCHEMA', ds.schema, flush=True)
    print('MAPS', len(ds.maps), flush=True)
    if not ds.maps:
        continue
    proj = ds.maps[0].embeddings.projected
    print('PROJECTED_COLS', list(proj.columns), flush=True)
    idcol = 'atlas_id' if 'atlas_id' in proj.columns else ('id' if 'id' in proj.columns else proj.columns[0])
    ids = [str(x) for x in proj[idcol].head(3).tolist()]
    rows = ds.get_data(ids=ids)
    for i,row in enumerate(rows):
        safe = {k:v for k,v in row.items() if k not in ('embedding','_embeddings')}
        print('ROW', i, json.dumps(safe, sort_keys=True, default=str), flush=True)
