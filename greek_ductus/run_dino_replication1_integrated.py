#!/usr/bin/env python3
from __future__ import annotations
import io, json, math, os, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import requests
from PIL import Image
import extract_cpu as ec

MODEL='facebook/dinov3-vitb16-pretrain-lvd1689m'
REV='5931719e67bbdb9737e363e781fb0c67687896bc'
NPERM=20000
SEED=411
BATCH=64
ROOT=Path(__file__).parent
UA={'User-Agent':'voynich-greek-ductus-dino/2026-09-01'}


def retry_bytes(url,label,attempts=6):
    last=None
    for k in range(attempts):
        try:
            r=requests.get(url,headers=UA,timeout=90); r.raise_for_status(); return r.content
        except Exception as e:
            last=e; print(json.dumps({'event':'retry','label':label,'attempt':k+1,'error':repr(e)}),flush=True); time.sleep(min(10,1.5**k))
    raise RuntimeError(f'GET failed {label}: {last!r}')

def retry_json(url,label): return json.loads(retry_bytes(url,label).decode('utf-8'))
def retry_image(url,label): return Image.open(io.BytesIO(retry_bytes(url,label))).convert('RGB')
def fixed_indices(n,fracs): return sorted(set(max(0,min(n-1,int(round(f*(n-1))))) for f in fracs))
def even_take(items,n):
    if len(items)<n: raise RuntimeError(f'need {n} items, have {len(items)}')
    ix=np.linspace(0,len(items)-1,n).round().astype(int); return [items[int(i)] for i in ix]
def unit(x):
    x=np.asarray(x,np.float32); return x/max(float(np.linalg.norm(x)),1e-12)
def centroid(X): return unit(np.mean(X,axis=0))
def cosdist(a,b): return float(1-np.dot(a,b))

def page_crops(manifest,inds,shelf,fam):
    cs=ec.canvases(retry_json(manifest,f'manifest:{shelf}')); out=[]
    for pi in inds:
        im=retry_image(ec.image_url(cs[pi]),f'image:{shelf}:{pi}'); bw=ec.ink_mask(im); boxes=ec.word_boxes(bw)
        for ci,b in enumerate(boxes): out.append((ec.norm_crop(bw,b),{'family':fam,'shelfmark':shelf,'page':int(pi),'crop_index':int(ci)}))
        print(json.dumps({'event':'crop_page','family':fam,'shelfmark':shelf,'page':pi,'crops':len(boxes)}),flush=True)
    return out

def main():
    import torch
    from transformers import AutoImageProcessor, AutoModel
    token=os.getenv('HF_TOKEN')
    spec=json.loads((ROOT/'replication1_manifest.json').read_text())
    images=[]; meta=[]
    for fam,rows in spec['families'].items():
        for r in rows:
            cs=ec.canvases(retry_json(r['manifest'],f'index:{r["shelfmark"]}')); inds=fixed_indices(len(cs),spec['control_page_fractions']); items=page_crops(r['manifest'],inds,r['shelfmark'],fam); chosen=even_take(items,90)
            for im,m in chosen: images.append(im); meta.append(m)
            print(json.dumps({'event':'crop_ms','family':fam,'shelfmark':r['shelfmark'],'available':len(items),'selected':90}),flush=True)
    vcs=ec.canvases(retry_json(spec['vms_manifest'],'vms_manifest')); vinds=fixed_indices(len(vcs),spec['vms_fractions']); items=page_crops(spec['vms_manifest'],vinds,'Beinecke408','VOYNICH'); chosen=even_take(items,270)
    for im,m in chosen: images.append(im); meta.append(m)
    assert len(images)==1710
    print(json.dumps({'event':'crop_bank_ready','n':len(images)}),flush=True)

    proc=AutoImageProcessor.from_pretrained(MODEL,revision=REV,token=token)
    model=AutoModel.from_pretrained(MODEL,revision=REV,token=token,torch_dtype=torch.float16,low_cpu_mem_usage=True).eval().cuda()
    em=[]
    with torch.inference_mode():
        for i in range(0,len(images),BATCH):
            x=proc(images=images[i:i+BATCH],return_tensors='pt').to('cuda')
            with torch.autocast(device_type='cuda',dtype=torch.float16): y=model(**x).last_hidden_state[:,0].float()
            y=torch.nn.functional.normalize(y,dim=1); em.append(y.cpu().numpy())
            print(json.dumps({'event':'dino_batch','done':min(i+BATCH,len(images)),'total':len(images)}),flush=True)
    E=np.concatenate(em).astype(np.float32)

    by_ms=defaultdict(list); by_page=defaultdict(list); vix=[]
    for i,m in enumerate(meta):
        if m['family']=='VOYNICH': vix.append(i); by_page[int(m['page'])].append(i)
        else: by_ms[(m['family'],m['shelfmark'])].append(i)
    ms=[]
    for (fam,shelf),ix in sorted(by_ms.items()): ms.append({'family':fam,'shelfmark':shelf,'centroid':centroid(E[ix]),'n':len(ix)})
    M=np.stack([r['centroid'] for r in ms]); labs=np.array([r['family'] for r in ms],dtype=object); v=centroid(E[vix]); cg=centroid(M[labs=='GREEK']); cl=centroid(M[labs=='LATIN'])
    dg=cosdist(v,cg); dl=cosdist(v,cl); A=dl-dg
    rng=np.random.default_rng(SEED); null=np.empty(NPERM,np.float32)
    for k in range(NPERM):
        lp=rng.permutation(labs); null[k]=cosdist(v,centroid(M[lp=='LATIN']))-cosdist(v,centroid(M[lp=='GREEK']))
    mu=float(null.mean()); sd=float(null.std(ddof=1)); z=float((A-mu)/sd); p=float((1+(null>=A).sum())/(NPERM+1))
    good=0; confusion={'GREEK':{'GREEK':0,'LATIN':0},'LATIN':{'GREEK':0,'LATIN':0}}; per=[]
    for i,r in enumerate(ms):
        keep=np.arange(len(ms))!=i; lm=labs[keep]; mm=M[keep]; g=centroid(mm[lm=='GREEK']); l=centroid(mm[lm=='LATIN']); d={'GREEK':cosdist(M[i],g),'LATIN':cosdist(M[i],l)}; pred=min(d,key=d.get); good+=pred==r['family']; confusion[r['family']][pred]+=1; per.append({'family':r['family'],'shelfmark':r['shelfmark'],'n':r['n'],'loo_pred':pred,'distances':d})
    pages=[]
    for pg,ix in sorted(by_page.items()):
        q=centroid(E[ix]); a=cosdist(q,cl)-cosdist(q,cg); pages.append({'page':pg,'n':len(ix),'greek_advantage':a})
    result={'protocol':'2026-09-01.greek-ductus.dino-replication1','model':MODEL,'revision':REV,'n_embeddings':len(E),'embedding_dim':int(E.shape[1]),'distances':{'GREEK':dg,'LATIN':dl},'greek_advantage':float(A),'perm_mean':mu,'perm_sd':sd,'z':z,'empirical_p_one_sided':p,'permutations':NPERM,'seed':SEED,'loo_accuracy':good/len(ms),'loo_confusion':confusion,'per_manuscript':per,'vms_pages':pages,'vms_fraction_positive':float(np.mean([x['greek_advantage']>0 for x in pages])),'decision':'DINO_PASS_Z_GE_2' if z>=2 else ('DINO_UNRESOLVED_1_TO_2' if z>=1 else 'DINO_NOT_SUPPORTIVE_Z_LT_1')}
    print('DINO_RESULT='+json.dumps(result,separators=(',',':')),flush=True)

if __name__=='__main__': main()
