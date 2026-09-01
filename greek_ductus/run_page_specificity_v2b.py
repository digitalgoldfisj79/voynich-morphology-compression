#!/usr/bin/env python3
from __future__ import annotations
import argparse,io,urllib.request
from PIL import Image
import run_page_specificity_v2 as v2

# Execution-layer patch only: Heidelberg serves canonical images to urllib but rejects
# the shared requests/User-Agent path. Scientific targets and all downstream code remain unchanged.
def urllib_fetch(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return Image.open(io.BytesIO(r.read())).convert('RGB')

def patched_fetch_canvas_image(c,oid,lab):
    last=None
    for u in v2.canonical_image_urls(c):
        try:
            if 'digi.ub.uni-heidelberg.de' in u:
                return urllib_fetch(u)
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
