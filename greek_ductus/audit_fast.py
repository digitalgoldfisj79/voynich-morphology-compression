#!/usr/bin/env python3
"""Execution-only wrapper for extract_cpu QC.

Same frozen segmentation functions; only network timeouts, concurrency and progress
logging differ from extract_cpu.py. No scientific metric is computed here.
"""
from __future__ import annotations
import base64, io, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import requests
from PIL import Image
import extract_cpu as ec

TIMEOUT_JSON=20
TIMEOUT_IMAGE=30
WORKERS=6

def get_json_fast(url):
    r=requests.get(url,headers=ec.UA,timeout=TIMEOUT_JSON); r.raise_for_status(); return r.json()

def fetch_image_fast(url):
    r=requests.get(url,headers=ec.UA,timeout=TIMEOUT_IMAGE); r.raise_for_status(); return Image.open(io.BytesIO(r.content)).convert('RGB')

ec.get_json=get_json_fast
ec.fetch_image=fetch_image_fast

def one(fam,rec):
    cs=ec.canvases(ec.get_json(rec['manifest']))
    inds=ec.choose_indices(len(cs),True); counts=[]; dims=[]; items=[]
    for ci in inds:
        im=ec.fetch_image(ec.image_url(cs[ci])); bw=ec.ink_mask(im); bs=ec.word_boxes(bw); counts.append(len(bs))
        for b in bs[:20]:
            nc=ec.norm_crop(bw,b); dims.append([b[2],b[3]]); items.append((nc,f"{rec['shelfmark']} p{ci}"))
    return fam,rec['shelfmark'],{
        'shelfmark':rec['shelfmark'],'canvases':len(cs),'sampled_indices':inds,'crop_counts':counts,
        'total_crops':sum(counts),'median_box_wh':([float(np.median([d[0] for d in dims])),float(np.median([d[1] for d in dims]))] if dims else None)
    },items

def main():
    spec=json.loads(Path(__file__).with_name('primary_manifest.json').read_text())
    summary={'protocol':spec['protocol'],'audit':True,'execution':'bounded_parallel_v1','families':{f:{'manuscripts':[]} for f in spec['families']}}
    sheets={f:[] for f in spec['families']}
    futs=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fam,recs in spec['families'].items():
            for rec in recs: futs.append(ex.submit(one,fam,rec))
        done=0
        for fut in as_completed(futs):
            done+=1
            try:
                fam,shelf,r,items=fut.result(); summary['families'][fam]['manuscripts'].append(r); sheets[fam].extend(items)
                print(json.dumps({'event':'qc_manuscript','done':done,'total':len(futs),'family':fam,'shelfmark':shelf,'crops':r['total_crops']}),flush=True)
            except Exception as e:
                print(json.dumps({'event':'qc_error','done':done,'total':len(futs),'error':f'{type(e).__name__}: {e}'}),flush=True)
    for fam in summary['families']:
        rr=sorted(summary['families'][fam]['manuscripts'],key=lambda x:x['shelfmark']); summary['families'][fam]['manuscripts']=rr
        summary['families'][fam]['successful']=len(rr); summary['families'][fam]['total_crops']=sum(r.get('total_crops',0) for r in rr)
    print('QC_SUMMARY='+json.dumps(summary,separators=(',',':'),ensure_ascii=False),flush=True)
    for fam,data in sheets.items():
        if data:
            jpg=ec.contact_sheet(data,limit=36)
            print(f'CONTACT_{fam}_B64='+base64.b64encode(jpg).decode(),flush=True)
if __name__=='__main__': main()
