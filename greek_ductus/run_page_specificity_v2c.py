#!/usr/bin/env python3
from __future__ import annotations
import argparse,io,urllib.request
from PIL import Image
import run_page_specificity_v2 as v2

# Acquisition-only wrapper. For Heidelberg, bypass IIIF derivatives entirely and
# use the manifest's direct JPEG resource.@id, which was preflighted HTTP 200 for
# every frozen target/control page. Scientific targets/features/decision rules are unchanged.
def urllib_fetch(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return Image.open(io.BytesIO(r.read())).convert('RGB')

def patched_fetch_canvas_image(c,oid,lab):
    try:
        r=c['images'][0]['resource']
        direct=r.get('@id') or r.get('id')
    except Exception:
        direct=None
    if direct and 'digi.ub.uni-heidelberg.de' in direct:
        return urllib_fetch(direct)
    last=None
    for u in v2.canonical_image_urls(c):
        try:
            return v2.base.fetch_img(u)
        except Exception as e:
            last=e
            print({'event':'image_fallback','object':oid,'folio':lab,'url':u,'error':repr(e)},flush=True)
    raise last or RuntimeError(f'{oid} {lab}: no image URL')

v2.fetch_canvas_image=patched_fetch_canvas_image

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['cpu','dino'],required=True); a=ap.parse_args()
    objects,meta=v2.collect()
    v2.cpu_run(objects,meta) if a.mode=='cpu' else v2.dino_run(objects,meta)
