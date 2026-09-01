#!/usr/bin/env python3
from __future__ import annotations
import json, math, time
from collections import Counter
from pathlib import Path
import numpy as np
import extract_cpu as ec
import stroke_features as sf

ROOT=Path(__file__).parent
SEED=409
NPERM=20000


def summ(ff):
    x=np.vstack(ff).astype(float)
    return np.r_[np.median(x,0), np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)]


def scale(C):
    m=np.median(C,0)
    d=np.median(np.abs(C-m),0)*1.4826
    d=np.where(d<1e-9,1.,d)
    return m,d


def dst(a,b):
    return float(np.linalg.norm(a-b)/math.sqrt(len(a)))


def fixed_indices(n,fracs):
    return sorted(set(max(0,min(n-1,int(round(f*(n-1))))) for f in fracs))


def page_features(im):
    bw=ec.ink_mask(im)
    bs=ec.word_boxes(bw)
    return [sf.descriptor(ec.norm_crop(bw,b)) for b in bs]


def analyze(records,qraw):
    C=np.vstack([r['summary'] for r in records])
    m,d=scale(C)
    X=(C-m)/d
    q=(qraw-m)/d
    lab=np.array([r['family'] for r in records],dtype=object)
    fs=['GREEK','LATIN']
    cen={f:np.median(X[lab==f],0) for f in fs}
    ds={f:dst(q,cen[f]) for f in fs}
    A=ds['LATIN']-ds['GREEK']
    rng=np.random.default_rng(SEED)
    null=[]
    for _ in range(NPERM):
        lp=rng.permutation(lab)
        cc={f:np.median(X[lp==f],0) for f in fs}
        dd={f:dst(q,cc[f]) for f in fs}
        null.append(dd['LATIN']-dd['GREEK'])
    null=np.asarray(null)
    mu=float(null.mean()); sd=float(null.std(ddof=1))
    z=float((A-mu)/sd)
    p=float((1+(null>=A).sum())/(len(null)+1))
    per=[]; good=0; conf={f:{g:0 for g in fs} for f in fs}
    for i,r in enumerate(records):
        cc={}
        for f in fs:
            ix=np.where(lab==f)[0]
            ix=ix[ix!=i]
            cc[f]=np.median(X[ix],0)
        dd={f:dst(X[i],cc[f]) for f in fs}
        pred=min(dd,key=dd.get)
        good+=pred==r['family']; conf[r['family']][pred]+=1
        per.append({'family':r['family'],'shelfmark':r['shelfmark'],'n_crops':r['n_crops'],'distances':{f:dst(X[i],cen[f]) for f in fs},'loo_pred':pred})
    return {'distances':ds,'greek_advantage':float(A),'perm_mean':mu,'perm_sd':sd,'z':z,'empirical_p_one_sided':p,'loo_accuracy':good/len(records),'loo_confusion':conf,'per_manuscript':per,'scaler_center':m,'scaler_scale':d,'centroids':cen}


def main():
    spec=json.loads((ROOT/'replication1_manifest.json').read_text())
    t=time.time(); recs=[]; counts=[]
    for fam,rr in spec['families'].items():
        for r in rr:
            cs=ec.canvases(ec.get_json(r['manifest']))
            inds=fixed_indices(len(cs),spec['control_page_fractions'])
            ff=[]
            for pi in inds:
                im=ec.fetch_image(ec.image_url(cs[pi]))
                f=page_features(im); ff.extend(f)
                counts.append({'family':fam,'shelfmark':r['shelfmark'],'page':pi,'crops':len(f)})
                print(json.dumps({'event':'rep_page','family':fam,'shelfmark':r['shelfmark'],'page':pi,'crops':len(f),'elapsed_s':round(time.time()-t,1)}),flush=True)
            recs.append({'family':fam,'shelfmark':r['shelfmark'],'summary':summ(ff),'n_crops':len(ff)})
            print(json.dumps({'event':'rep_ms','family':fam,'shelfmark':r['shelfmark'],'crops':len(ff)}),flush=True)

    vm=ec.get_json(spec['vms_manifest']); vcs=ec.canvases(vm)
    vf=[]; vp=[]
    for pi in fixed_indices(len(vcs),spec['vms_fractions']):
        im=ec.fetch_image(ec.image_url(vcs[pi])); f=page_features(im); vf.extend(f)
        vp.append({'page':pi,'crops':len(f),'summary':summ(f) if f else None})
        print(json.dumps({'event':'rep_vms_page','page':pi,'crops':len(f),'elapsed_s':round(time.time()-t,1)}),flush=True)

    q=summ(vf)
    result=analyze(recs,q)
    # page-level descriptive spread using the control scaler/centroids from the headline fit
    m=result.pop('scaler_center'); d=result.pop('scaler_scale'); cen=result.pop('centroids')
    pages=[]
    for p in vp:
        if p['summary'] is None: continue
        x=(p['summary']-m)/d
        dg=dst(x,cen['GREEK']); dl=dst(x,cen['LATIN'])
        pages.append({'page':p['page'],'crops':p['crops'],'greek_distance':dg,'latin_distance':dl,'greek_advantage':dl-dg})
    adv=[p['greek_advantage'] for p in pages]
    result.update({
        'protocol':spec['protocol'],
        'feature_version':sf.FEATURE_VERSION,
        'permutations':NPERM,
        'seed':SEED,
        'counts':{'controls':len(recs),'family_counts':dict(Counter(r['family'] for r in recs)),'control_crops':sum(r['n_crops'] for r in recs),'vms_pages':len(vp),'vms_crops':len(vf)},
        'page_counts':counts,
        'vms_pages':pages,
        'vms_page_advantage_median':float(np.median(adv)),
        'vms_page_advantage_iqr':float(np.quantile(adv,.75)-np.quantile(adv,.25)),
        'vms_fraction_positive':float(np.mean(np.asarray(adv)>0)),
        'decision':'REPLICATION_PASS_Z_GE_2' if result['z']>=2 else ('REPLICATION_UNRESOLVED_1_TO_2' if result['z']>=1 else 'REPLICATION_NOT_SUPPORTIVE_Z_LT_1')
    })
    print('REPLICATION1_RESULT='+json.dumps(result,separators=(',',':'),ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
