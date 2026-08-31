#!/usr/bin/env python3
from __future__ import annotations
import json,math,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import extract_cpu as ec
import stroke_features as sf

ROOT=Path(__file__).parent; SEED=408; NPERM=20000

def summ(ff):
 x=np.vstack(ff).astype(float); return np.r_[np.median(x,0),np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)]
def scale(C):
 m=np.median(C,0); d=np.median(np.abs(C-m),0)*1.4826; d=np.where(d<1e-9,1.,d); return m,d
def dst(a,b): return float(np.linalg.norm(a-b)/math.sqrt(len(a)))
def analyze(records,qraw):
 C=np.vstack([r['summary'] for r in records]); m,d=scale(C); X=(C-m)/d; q=(qraw-m)/d
 lab=np.array([r['family'] for r in records],dtype=object); fs=['GREEK','ITALIAN_LATIN','GERMAN']
 cen={f:np.median(X[lab==f],0) for f in fs}; ds={f:dst(q,cen[f]) for f in fs}
 A=.5*(ds['ITALIAN_LATIN']+ds['GERMAN'])-ds['GREEK']; rng=np.random.default_rng(SEED); null=[]
 for _ in range(NPERM):
  lp=rng.permutation(lab); cc={f:np.median(X[lp==f],0) for f in fs}; dd={f:dst(q,cc[f]) for f in fs}; null.append(.5*(dd['ITALIAN_LATIN']+dd['GERMAN'])-dd['GREEK'])
 null=np.asarray(null); mu=float(null.mean()); sd=float(null.std(ddof=1)); z=float((A-mu)/sd); p=float((1+(null>=A).sum())/(len(null)+1))
 pair={f'{a}__{b}':dst(cen[a],cen[b]) for i,a in enumerate(fs) for b in fs[i+1:]}
 per=[{'family':r['family'],'shelfmark':r['shelfmark'],'host':r['host'],'n_crops':r['n_crops'],'distances':{f:dst(X[i],cen[f]) for f in fs}} for i,r in enumerate(records)]
 conf={f:{g:0 for g in fs} for f in fs}; good=0
 for i,r in enumerate(records):
  cc={};
  for f in fs:
   ix=np.where(lab==f)[0]; ix=ix[ix!=i]; cc[f]=np.median(X[ix],0)
  pred=min(cc,key=lambda f:dst(X[i],cc[f])); conf[r['family']][pred]+=1; good+=pred==r['family']
 return {'distances':ds,'greek_advantage':float(A),'perm_mean':mu,'perm_sd':sd,'z':z,'empirical_p_one_sided':p,'pairwise_centroids':pair,'loo_accuracy':good/len(records),'loo_confusion':conf,'per_manuscript':per}

def page_features(im):
 bw=ec.ink_mask(im); bs=ec.word_boxes(bw); return [sf.descriptor(ec.norm_crop(bw,b)) for b in bs]
def indices(n,fracs): return sorted(set(max(0,min(n-1,int(round(f*(n-1))))) for f in fracs))
def main():
 spec=json.loads((ROOT/'operational_manifest.json').read_text()); vs=json.loads((ROOT/'vms_manifest.json').read_text()); t=time.time(); recs=[]; page_counts=[]
 for fam,rr in spec['families'].items():
  for r in rr:
   cs=ec.canvases(ec.get_json(r['manifest'])); ff=[]; inds=ec.choose_indices(len(cs),False)
   for pi in inds:
    im=ec.fetch_image(ec.image_url(cs[pi])); f=page_features(im); ff.extend(f); page_counts.append({'family':fam,'shelfmark':r['shelfmark'],'page':pi,'crops':len(f)}); print(json.dumps({'event':'page','family':fam,'shelfmark':r['shelfmark'],'page':pi,'crops':len(f),'elapsed_s':round(time.time()-t,1)}),flush=True)
   recs.append({'family':fam,'shelfmark':r['shelfmark'],'host':r['host'],'summary':summ(ff),'n_crops':len(ff)})
   print(json.dumps({'event':'manuscript','family':fam,'shelfmark':r['shelfmark'],'crops':len(ff)}),flush=True)
 m=ec.get_json(vs['manifest']); cs=ec.canvases(m); vf=[]; vpages=[]
 for pi in indices(len(cs),vs['fractions']):
  im=ec.fetch_image(ec.image_url(cs[pi])); f=page_features(im); vf.extend(f); vpages.append({'page':pi,'crops':len(f),'summary':summ(f) if f else None}); print(json.dumps({'event':'vms_page','page':pi,'crops':len(f),'elapsed_s':round(time.time()-t,1)}),flush=True)
 q=summ(vf); result=analyze(recs,q); result.update({'protocol':spec['protocol'],'feature_version':sf.FEATURE_VERSION,'permutations':NPERM,'seed':SEED,'counts':{'controls':len(recs),'control_crops':sum(r['n_crops'] for r in recs),'vms_pages':len(vpages),'vms_crops':len(vf),'family_counts':dict(Counter(r['family'] for r in recs)),'host_by_family':{f:dict(Counter(r['host'] for r in recs if r['family']==f)) for f in ['GREEK','ITALIAN_LATIN','GERMAN']}},'page_counts':page_counts,'vms_page_counts':[{'page':p['page'],'crops':p['crops']} for p in vpages]})
 result['decision']='CPU_PASS_Z_GE_2' if result['z']>=2 else ('CPU_STOP_RECOMMENDED_Z_LT_1' if result['z']<1 else 'CPU_UNRESOLVED_1_TO_2')
 print('CPU_PRIMARY_RESULT='+json.dumps(result,separators=(',',':'),ensure_ascii=False),flush=True)
if __name__=='__main__': main()
