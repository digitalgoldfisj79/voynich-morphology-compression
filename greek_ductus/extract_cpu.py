#!/usr/bin/env python3
"""Deterministic IIIF text-crop extraction for the Greek-ductus experiment.

No labels or transcription are used. The same page sampling, binarisation and
component grouping is applied to all ordinary-script families.
"""
from __future__ import annotations
import argparse, base64, hashlib, io, json, math, random, statistics
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps

SEED=408
FRACTIONS=(0.16,0.30,0.44,0.58,0.72,0.86)
UA={"User-Agent":"voynich-greek-ductus/2026-08-31"}

def get_json(url:str)->dict[str,Any]:
    r=requests.get(url,headers=UA,timeout=60); r.raise_for_status(); return r.json()

def canvases(m:dict[str,Any])->list[dict[str,Any]]:
    out=[]
    if "items" in m: # IIIF v3
        for i,c in enumerate(m.get("items",[])):
            try:
                body=c["items"][0]["items"][0]["body"]
                services=body.get("service") or []
                if isinstance(services,dict): services=[services]
                sid=(services[0].get("id") or services[0].get("@id")) if services else None
                bid=body.get("id") or body.get("@id")
                out.append({"i":i,"service":sid,"body":bid,"label":c.get("label")})
            except Exception: pass
    else: # IIIF v2
        for i,c in enumerate((m.get("sequences") or [{}])[0].get("canvases",[])):
            try:
                res=c["images"][0]["resource"]
                srv=res.get("service") or {}
                if isinstance(srv,list): srv=srv[0] if srv else {}
                out.append({"i":i,"service":srv.get("@id") or srv.get("id"),"body":res.get("@id") or res.get("id"),"label":c.get("label")})
            except Exception: pass
    return [x for x in out if x.get("service") or x.get("body")]

def image_url(c:dict[str,Any],width:int=1600)->str:
    sid=c.get("service")
    if sid: return sid.rstrip("/")+f"/full/{width},/0/default.jpg"
    u=c["body"]
    if "/full/" in u:
        pre=u.split("/full/")[0]; return pre+f"/full/{width},/0/default.jpg"
    return u

def choose_indices(n:int,audit:bool)->list[int]:
    fs=(0.50,) if audit else FRACTIONS
    return sorted(set(max(0,min(n-1,int(round(f*(n-1))))) for f in fs))

def fetch_image(url:str)->Image.Image:
    r=requests.get(url,headers=UA,timeout=90); r.raise_for_status(); return Image.open(io.BytesIO(r.content)).convert("RGB")

def ink_mask(im:Image.Image)->np.ndarray:
    a=np.array(im.convert("L"))
    # suppress page-edge darkness before adaptive thresholding
    h,w=a.shape; y0,y1=int(.035*h),int(.965*h); x0,x1=int(.035*w),int(.965*w)
    a=a[y0:y1,x0:x1]
    a=cv2.GaussianBlur(a,(3,3),0)
    block=max(31,((min(a.shape)//35)//2)*2+1)
    block=min(block,101); block=block if block%2 else block+1
    bw=cv2.adaptiveThreshold(a,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,17)
    # remove isolated dust but do not close characters together
    bw=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    return bw

def word_boxes(bw:np.ndarray)->list[tuple[int,int,int,int]]:
    H,W=bw.shape
    n,lab,st,_=cv2.connectedComponentsWithStats(bw,8)
    comps=[]
    for k in range(1,n):
        x,y,w,h,a=map(int,st[k])
        if a<7 or h<5 or h>0.075*H or w>0.10*W or w*h>0.008*W*H: continue
        if w<2 or a/max(1,w*h)>.75: continue
        comps.append([x,y,w,h,a,y+h/2,x+w/2])
    if len(comps)<20: return []
    mh=float(np.median([c[3] for c in comps])); line_tol=max(5,0.65*mh)
    comps.sort(key=lambda c:(c[5],c[0])); lines=[]
    for c in comps:
        best=None; bd=1e9
        for j,L in enumerate(lines):
            d=abs(c[5]-L["cy"])
            if d<line_tol and d<bd: best=j;bd=d
        if best is None: lines.append({"cy":c[5],"c":[c]})
        else:
            L=lines[best]; L["c"].append(c); L["cy"]=float(np.median([z[5] for z in L["c"]]))
    boxes=[]
    for L in lines:
        cc=sorted(L["c"],key=lambda c:c[0])
        if len(cc)<3: continue
        hmed=max(4,float(np.median([c[3] for c in cc])))
        # component gaps within a handwritten word are normally much smaller than line height
        gap_lim=0.42*hmed
        groups=[]; cur=[cc[0]]
        for c in cc[1:]:
            p=cur[-1]; gap=c[0]-(p[0]+p[2])
            if gap<=gap_lim: cur.append(c)
            else: groups.append(cur); cur=[c]
        groups.append(cur)
        for g in groups:
            x0=min(c[0] for c in g); y0=min(c[1] for c in g); x1=max(c[0]+c[2] for c in g); y1=max(c[1]+c[3] for c in g)
            w=x1-x0; h=y1-y0
            if h<7 or w<8 or h>0.08*H or w>0.25*W: continue
            ar=w/max(1,h)
            if ar<0.35 or ar>9.0: continue
            pad=max(2,int(round(.16*h))); x0=max(0,x0-pad);y0=max(0,y0-pad);x1=min(W,x1+pad);y1=min(H,y1+pad)
            patch=bw[y0:y1,x0:x1]; dens=float((patch>0).mean())
            if .025<=dens<=.55: boxes.append((x0,y0,x1-x0,y1-y0))
    # deterministic ordering and cap per page
    boxes=sorted(boxes,key=lambda b:(b[1],b[0],b[2],b[3]))
    return boxes[:90]

def norm_crop(bw:np.ndarray,b:tuple[int,int,int,int])->Image.Image:
    x,y,w,h=b; p=bw[y:y+h,x:x+w]
    ys,xs=np.where(p>0)
    if len(xs): p=p[max(0,ys.min()-2):min(p.shape[0],ys.max()+3),max(0,xs.min()-2):min(p.shape[1],xs.max()+3)]
    im=Image.fromarray(255-p).convert("L")
    target_h=64; scale=target_h/max(1,im.height); nw=max(6,min(240,int(round(im.width*scale))))
    im=im.resize((nw,target_h),Image.Resampling.LANCZOS)
    can=Image.new("L",(256,80),255); can.paste(im,((256-nw)//2,8)); return can.convert("RGB")

def contact_sheet(items:list[tuple[Image.Image,str]],limit:int=72)->bytes:
    items=items[:limit]; cellw,cellh=264,104; cols=6; rows=max(1,math.ceil(len(items)/cols))
    out=Image.new("RGB",(cols*cellw,rows*cellh),"white"); d=ImageDraw.Draw(out)
    for i,(im,label) in enumerate(items):
        x=(i%cols)*cellw;y=(i//cols)*cellh; out.paste(im,(x+4,y+4));d.text((x+4,y+86),label[:35],fill="black")
    q=io.BytesIO();out.save(q,"JPEG",quality=78,optimize=True);return q.getvalue()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--manifest",default=str(Path(__file__).with_name("primary_manifest.json")));ap.add_argument("--audit",action="store_true");args=ap.parse_args()
    spec=json.loads(Path(args.manifest).read_text()); rng=random.Random(SEED)
    summary={"protocol":spec["protocol"],"audit":args.audit,"families":{}}; sheets={}
    for fam,records in spec["families"].items():
        fam_items=[]; recs=[]
        for rec in records:
            try:
                cs=canvases(get_json(rec["manifest"])); inds=choose_indices(len(cs),args.audit); counts=[]; dims=[]
                for ci in inds:
                    im=fetch_image(image_url(cs[ci])); bw=ink_mask(im); bs=word_boxes(bw); counts.append(len(bs));
                    for j,b in enumerate(bs[:20] if args.audit else bs):
                        nc=norm_crop(bw,b); dims.append([b[2],b[3]])
                        if len(fam_items)<120: fam_items.append((nc,f"{rec['shelfmark']} p{ci}"))
                recs.append({"shelfmark":rec["shelfmark"],"canvases":len(cs),"sampled_indices":inds,"crop_counts":counts,"total_crops":sum(counts),"median_box_wh":([float(np.median([d[0] for d in dims])),float(np.median([d[1] for d in dims]))] if dims else None)})
            except Exception as e: recs.append({"shelfmark":rec["shelfmark"],"error":f"{type(e).__name__}: {e}"})
        summary["families"][fam]={"manuscripts":recs,"successful":sum("error" not in r for r in recs),"total_crops":sum(r.get("total_crops",0) for r in recs)}
        sheets[fam]=contact_sheet(fam_items)
    print("QC_SUMMARY="+json.dumps(summary,separators=(",",":"),ensure_ascii=False),flush=True)
    for fam,data in sheets.items(): print(f"CONTACT_{fam}_B64="+base64.b64encode(data).decode(),flush=True)
if __name__=="__main__": main()
