#!/usr/bin/env python3
"""Frozen CPU sensitivity analyses for protocol 2026-08-31.greek-ductus.v1.

Implements only diagnostics preregistered in ANALYSIS_PLAN.md and
SENSITIVITY_PARAMETERS.md. These analyses cannot replace the primary result.
"""
from __future__ import annotations
import json, math, time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import extract_cpu as ec
import stroke_features as sf
import run_cpu_primary as rp

ROOT=Path(__file__).parent
SECURE_EAST={'Mscr.Dresd.Da.61','Mscr.Dresd.Da.47'}
FERRARA='Cod.graec. 256'


def ink_mask_c21(im:Image.Image)->np.ndarray:
    a=np.array(im.convert('L'))
    h,w=a.shape; y0,y1=int(.035*h),int(.965*h); x0,x1=int(.035*w),int(.965*w)
    a=a[y0:y1,x0:x1]
    a=cv2.GaussianBlur(a,(3,3),0)
    block=max(31,((min(a.shape)//35)//2)*2+1); block=min(block,101); block=block if block%2 else block+1
    return cv2.adaptiveThreshold(a,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,21)


def lowinfo(im:Image.Image)->bool:
    m=np.array(im.convert('L'))<180
    if not m.any(): return True
    s=sf._skeleton(m); deg=sf._degree(s)
    endpoints=int(np.logical_and(s>0,deg==1).sum()); junctions=int(np.logical_and(s>0,deg>=3).sum())
    ys,xs=np.where(m); ar=float((xs.max()-xs.min()+1)/max(1,ys.max()-ys.min()+1))
    comps=cv2.connectedComponents(m.astype(np.uint8),8)[0]-1
    from skimage.measure import euler_number
    holes=max(0,int(comps-euler_number(m,connectivity=2)))
    return (endpoints<=2 and junctions==0 and ar<.55) or (holes>=1 and .75<=ar<=1.33)


def collect(maskfun=ec.ink_mask, exclude_low=False):
    spec=json.loads((ROOT/'operational_manifest.json').read_text()); vs=json.loads((ROOT/'vms_manifest.json').read_text())
    recs=[]; vpages=[]; t=time.time()
    for fam,rr in spec['families'].items():
        for r in rr:
            cs=ec.canvases(ec.get_json(r['manifest'])); ff=[]
            for pi in ec.choose_indices(len(cs),False):
                im=ec.fetch_image(ec.image_url(cs[pi])); bw=maskfun(im); bs=ec.word_boxes(bw)
                for b in bs:
                    nc=ec.norm_crop(bw,b)
                    if exclude_low and lowinfo(nc): continue
                    ff.append(sf.descriptor(nc))
            recs.append({'family':fam,'shelfmark':r['shelfmark'],'host':r['host'],'summary':rp.summ(ff),'n_crops':len(ff)})
            print(json.dumps({'event':'sens_ms','mode':'c21' if maskfun is ink_mask_c21 else ('lowinfo' if exclude_low else 'base'),'family':fam,'shelfmark':r['shelfmark'],'crops':len(ff),'elapsed_s':round(time.time()-t,1)}),flush=True)
    m=ec.get_json(vs['manifest']); cs=ec.canvases(m); vf=[]
    for pi in rp.indices(len(cs),vs['fractions']):
        im=ec.fetch_image(ec.image_url(cs[pi])); bw=maskfun(im); bs=ec.word_boxes(bw); ff=[]
        for b in bs:
            nc=ec.norm_crop(bw,b)
            if exclude_low and lowinfo(nc): continue
            ff.append(sf.descriptor(nc)); vf.append(ff[-1])
        if ff: vpages.append({'page':pi,'summary':rp.summ(ff),'n_crops':len(ff)})
    return recs,rp.summ(vf),vpages


def subset_result(recs,q,keep):
    x=[r for r in recs if keep(r)]
    fam=Counter(r['family'] for r in x)
    if any(fam[f]<2 for f in ['GREEK','ITALIAN_LATIN','GERMAN']): return {'underidentified':True,'family_counts':dict(fam)}
    # rp.analyze expects balanced family sizes under its label permutation. For unequal retained counts,
    # downsample deterministically to the minimum N per family by shelfmark solely for repository sensitivity.
    n=min(fam.values()); xx=[]
    for f in ['GREEK','ITALIAN_LATIN','GERMAN']:
        xx += sorted([r for r in x if r['family']==f],key=lambda r:r['shelfmark'])[:n]
    out=rp.analyze(xx,q); out['family_counts_before_balance']=dict(fam); out['balanced_n_per_family']=n
    return out


def page_spread(recs,qpages):
    C=np.vstack([r['summary'] for r in recs]); m,d=rp.scale(C); X=(C-m)/d; lab=np.array([r['family'] for r in recs],dtype=object)
    cen={f:np.median(X[lab==f],0) for f in ['GREEK','ITALIAN_LATIN','GERMAN']}
    rows=[]
    for p in qpages:
        q=(p['summary']-m)/d; ds={f:rp.dst(q,cen[f]) for f in cen}; A=.5*(ds['ITALIAN_LATIN']+ds['GERMAN'])-ds['GREEK']
        rows.append({'page':p['page'],'n_crops':p['n_crops'],'distances':ds,'greek_advantage':A})
    return {'pages':rows,'advantage_median':float(np.median([r['greek_advantage'] for r in rows])),'advantage_iqr':float(np.quantile([r['greek_advantage'] for r in rows],.75)-np.quantile([r['greek_advantage'] for r in rows],.25)),'fraction_positive':float(np.mean([r['greek_advantage']>0 for r in rows]))}


def secure_east_distance(recs,q):
    C=np.vstack([r['summary'] for r in recs]); m,d=rp.scale(C); qz=(q-m)/d
    east=np.vstack([(r['summary']-m)/d for r in recs if r['shelfmark'] in SECURE_EAST])
    return {'shelfmarks':sorted(SECURE_EAST),'distance':rp.dst(qz,np.median(east,0))}


def main():
    # Base is recomputed identically because primary job output was log-only; this also checks reproducibility.
    recs,q,vpages=collect()
    base=rp.analyze(recs,q)
    lowr,lowq,_=collect(exclude_low=True); low=rp.analyze(lowr,lowq)
    c21r,c21q,_=collect(maskfun=ink_mask_c21); c21=rp.analyze(c21r,c21q)
    out={
      'protocol':'2026-08-31.greek-ductus.v1',
      'primary_reproduction':base,
      'page_block_spread':page_spread(recs,vpages),
      'low_information_exclusion':low,
      'strict_C21':c21,
      'no_dresden':subset_result(recs,q,lambda r:r['host']!='dresden'),
      'no_leipzig':subset_result(recs,q,lambda r:r['host']!='leipzig'),
      'exclude_ferrara':subset_result(recs,q,lambda r:r['shelfmark']!=FERRARA),
      'secure_east_descriptive':secure_east_distance(recs,q),
      'note':'Sensitivity analyses are diagnostic only and cannot rescue/replace primary Z=1.805 result.'
    }
    print('CPU_SENSITIVITY_RESULT='+json.dumps(out,separators=(',',':'),ensure_ascii=False),flush=True)
if __name__=='__main__': main()
