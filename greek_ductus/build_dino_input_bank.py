#!/usr/bin/env python3
from __future__ import annotations
import hashlib, io, json, time
from pathlib import Path
import numpy as np
import requests
from PIL import Image
from huggingface_hub import HfApi
import extract_cpu as ec

ROOT=Path(__file__).parent
DATASET_REPO='Digitalgoldfish79/voynich-greek-ductus-dino'
SEED=410
UA={"User-Agent":"voynich-greek-ductus/2026-09-01"}


def retry_bytes(url,label,attempts=6):
    last=None
    for k in range(attempts):
        try:
            r=requests.get(url,headers=UA,timeout=90); r.raise_for_status(); return r.content
        except Exception as e:
            last=e
            print(json.dumps({'event':'retry','label':label,'attempt':k+1,'error':repr(e)}),flush=True)
            time.sleep(min(10,1.5**k))
    raise RuntimeError(f'GET failed {label}: {last!r}')


def retry_json(url,label):
    return json.loads(retry_bytes(url,label).decode('utf-8'))


def retry_image(url,label):
    return Image.open(io.BytesIO(retry_bytes(url,label))).convert('RGB')


def fixed_indices(n,fracs):
    return sorted(set(max(0,min(n-1,int(round(f*(n-1))))) for f in fracs))


def even_take(items,n):
    if len(items)<n: raise RuntimeError(f'need {n} items, have {len(items)}')
    idx=np.linspace(0,len(items)-1,n).round().astype(int)
    return [items[int(i)] for i in idx]


def fetch_page_crops(manifest, inds, shelfmark, family):
    cs=ec.canvases(retry_json(manifest,f'manifest:{shelfmark}')); all_items=[]
    for pi in inds:
        im=retry_image(ec.image_url(cs[pi]),f'image:{shelfmark}:{pi}'); bw=ec.ink_mask(im); boxes=ec.word_boxes(bw)
        for ci,b in enumerate(boxes):
            crop=np.array(ec.norm_crop(bw,b).convert('L'),dtype=np.uint8)
            all_items.append((crop,{'family':family,'shelfmark':shelfmark,'page':int(pi),'crop_index':int(ci)}))
        print(json.dumps({'event':'bank_page','family':family,'shelfmark':shelfmark,'page':pi,'crops':len(boxes)}),flush=True)
    return all_items


def main():
    spec=json.loads((ROOT/'replication1_manifest.json').read_text())
    images=[]; meta=[]
    for fam,rows in spec['families'].items():
        for r in rows:
            cs=ec.canvases(retry_json(r['manifest'],f'index_manifest:{r["shelfmark"]}'))
            inds=fixed_indices(len(cs),spec['control_page_fractions'])
            items=fetch_page_crops(r['manifest'],inds,r['shelfmark'],fam)
            chosen=even_take(items,90)
            for im,m in chosen: images.append(im); meta.append(m)
            print(json.dumps({'event':'bank_ms','family':fam,'shelfmark':r['shelfmark'],'available':len(items),'selected':90}),flush=True)

    vcs=ec.canvases(retry_json(spec['vms_manifest'],'vms_manifest'))
    vinds=fixed_indices(len(vcs),spec['vms_fractions'])
    vitems=fetch_page_crops(spec['vms_manifest'],vinds,'Beinecke408','VOYNICH')
    chosen=even_take(vitems,270)
    for im,m in chosen: images.append(im); meta.append(m)

    A=np.stack(images).astype(np.uint8)
    meta_json=np.array([json.dumps(m,separators=(',',':'),sort_keys=True) for m in meta],dtype='U256')
    out=Path('/tmp/dino_input_bank.npz'); np.savez_compressed(out,images=A,meta=meta_json)
    sha=hashlib.sha256(out.read_bytes()).hexdigest()
    audit={'protocol':'2026-09-01.greek-ductus.dino-replication1','input_sha256':sha,'shape':list(A.shape),'n':len(A),'controls':sum(m['family']!='VOYNICH' for m in meta),'voynich':sum(m['family']=='VOYNICH' for m in meta),'families':{f:sum(m['family']==f for m in meta) for f in ['GREEK','LATIN','VOYNICH']},'source_manifest':'replication1_manifest.json'}
    ap=HfApi(); ap.create_repo(DATASET_REPO,repo_type='dataset',private=True,exist_ok=True)
    ap.upload_file(path_or_fileobj=str(out),path_in_repo='dino_input_bank.npz',repo_id=DATASET_REPO,repo_type='dataset',commit_message='Freeze DINO input bank')
    ap.upload_file(path_or_fileobj=json.dumps(audit,indent=2).encode(),path_in_repo='dino_input_audit.json',repo_id=DATASET_REPO,repo_type='dataset',commit_message='Add DINO input audit')
    print('DINO_INPUT_AUDIT='+json.dumps(audit,separators=(',',':')),flush=True)

if __name__=='__main__': main()
