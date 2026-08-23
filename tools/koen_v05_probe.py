#!/usr/bin/env python3
import json, os, subprocess, sys

IDENT = 'edwardbozzard/voynich-dinov3-proposals-100k-enriched-20260713-v2'
print(json.dumps({'nomic_secret_present': bool(os.getenv('NOMIC_API_KEY')), 'hf_secret_present': bool(os.getenv('HF_TOKEN'))}), flush=True)
key = os.getenv('NOMIC_API_KEY','')
if not key:
    raise SystemExit('NOMIC_API_KEY secret not present')
subprocess.run([sys.executable, '-m', 'nomic.cli', 'login', key], check=True, stdout=subprocess.DEVNULL)
from nomic import AtlasDataset

ds = AtlasDataset(identifier=IDENT)
print('IDENT', ds.identifier, flush=True)
print('TOTAL', int(ds.total_datums), flush=True)
print('META_KEYS', sorted(ds.meta.keys()), flush=True)
print('SCHEMA', ds.schema, flush=True)
print('MAPS', len(ds.maps), flush=True)
if not ds.maps:
    raise SystemExit('dataset has no maps')
proj = ds.maps[0].embeddings.projected
print('PROJECTED_TYPE', type(proj).__name__, flush=True)
print('PROJECTED_COLS', list(proj.columns), flush=True)
print('PROJECTED_HEAD', proj.head(5).to_json(orient='records'), flush=True)
idcol = 'id' if 'id' in proj.columns else proj.columns[0]
ids = [str(x) for x in proj[idcol].head(5).tolist()]
rows = ds.get_data(ids=ids)
print('ROW_COUNT', len(rows), flush=True)
for i,row in enumerate(rows):
    safe = {k:v for k,v in row.items() if k not in ('embedding','_embeddings')}
    print('ROW', i, json.dumps(safe, sort_keys=True, default=str), flush=True)
