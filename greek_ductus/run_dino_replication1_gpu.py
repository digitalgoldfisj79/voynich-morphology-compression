#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download

MODEL='facebook/dinov3-vitb16-pretrain-lvd1689m'
REV='5931719e67bbdb9737e363e781fb0c67687896bc'
DATASET_REPO='Digitalgoldfish79/voynich-greek-ductus-dino'
NPERM=20000
SEED=411
BATCH=64


def unit(x):
    x=np.asarray(x,dtype=np.float32); n=np.linalg.norm(x)
    return x/max(float(n),1e-12)

def cosdist(a,b): return float(1.0-np.dot(a,b))

def centroid(X): return unit(np.mean(X,axis=0))

def main():
    import torch
    from transformers import AutoImageProcessor, AutoModel
    token=os.getenv('HF_TOKEN')
    bank_path=hf_hub_download(DATASET_REPO,'dino_input_bank.npz',repo_type='dataset',token=token)
    bank_bytes=Path(bank_path).read_bytes(); input_sha=hashlib.sha256(bank_bytes).hexdigest()
    z=np.load(bank_path,allow_pickle=False)
    imgs=z['images']; meta=[json.loads(s) for s in z['meta'].tolist()]
    assert len(imgs)==1710 and sum(m['family']=='VOYNICH' for m in meta)==270

    proc=AutoImageProcessor.from_pretrained(MODEL,revision=REV,token=token)
    model=AutoModel.from_pretrained(MODEL,revision=REV,token=token,torch_dtype=torch.float16,low_cpu_mem_usage=True).eval().cuda()
    em=[]
    with torch.inference_mode():
        for i in range(0,len(imgs),BATCH):
            pil=[Image.fromarray(a).convert('RGB') for a in imgs[i:i+BATCH]]
            x=proc(images=pil,return_tensors='pt').to('cuda')
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                y=model(**x).last_hidden_state[:,0].float()
            y=torch.nn.functional.normalize(y,dim=1)
            em.append(y.cpu().numpy())
            print(json.dumps({'event':'dino_batch','done':min(i+BATCH,len(imgs)),'total':len(imgs)}),flush=True)
    E=np.concatenate(em).astype(np.float32)

    by_ms=defaultdict(list); by_page=defaultdict(list); vix=[]
    for i,m in enumerate(meta):
        if m['family']=='VOYNICH':
            vix.append(i); by_page[int(m['page'])].append(i)
        else: by_ms[(m['family'],m['shelfmark'])].append(i)
    ms=[]
    for (fam,shelf),ix in sorted(by_ms.items()): ms.append({'family':fam,'shelfmark':shelf,'centroid':centroid(E[ix]),'n':len(ix)})
    v=centroid(E[vix]); labs=np.array([r['family'] for r in ms],dtype=object); M=np.stack([r['centroid'] for r in ms])
    cg=centroid(M[labs=='GREEK']); cl=centroid(M[labs=='LATIN'])
    dg=cosdist(v,cg); dl=cosdist(v,cl); A=dl-dg
    rng=np.random.default_rng(SEED); null=np.empty(NPERM,dtype=np.float32)
    for k in range(NPERM):
        lp=rng.permutation(labs); g=centroid(M[lp=='GREEK']); l=centroid(M[lp=='LATIN']); null[k]=cosdist(v,l)-cosdist(v,g)
    mu=float(null.mean()); sd=float(null.std(ddof=1)); zz=float((A-mu)/sd); p=float((1+(null>=A).sum())/(NPERM+1))

    good=0; confusion={'GREEK':{'GREEK':0,'LATIN':0},'LATIN':{'GREEK':0,'LATIN':0}}; per=[]
    for i,r in enumerate(ms):
        keep=np.arange(len(ms))!=i; lm=labs[keep]; mm=M[keep]
        g=centroid(mm[lm=='GREEK']); l=centroid(mm[lm=='LATIN']); d={'GREEK':cosdist(M[i],g),'LATIN':cosdist(M[i],l)}; pred=min(d,key=d.get)
        good+=pred==r['family']; confusion[r['family']][pred]+=1; per.append({'family':r['family'],'shelfmark':r['shelfmark'],'n':r['n'],'loo_pred':pred,'distances':d})
    pages=[]
    for page,ix in sorted(by_page.items()):
        q=centroid(E[ix]); pdg=cosdist(q,cg); pdl=cosdist(q,cl); pages.append({'page':page,'n':len(ix),'greek_distance':pdg,'latin_distance':pdl,'greek_advantage':pdl-pdg})

    result={'protocol':'2026-09-01.greek-ductus.dino-replication1','model':MODEL,'revision':REV,'input_sha256':input_sha,'n_embeddings':len(E),'embedding_dim':int(E.shape[1]),'distances':{'GREEK':dg,'LATIN':dl},'greek_advantage':float(A),'perm_mean':mu,'perm_sd':sd,'z':zz,'empirical_p_one_sided':p,'permutations':NPERM,'seed':SEED,'loo_accuracy':good/len(ms),'loo_confusion':confusion,'per_manuscript':per,'vms_pages':pages,'vms_fraction_positive':float(np.mean([x['greek_advantage']>0 for x in pages])),'decision':'DINO_PASS_Z_GE_2' if zz>=2 else ('DINO_UNRESOLVED_1_TO_2' if zz>=1 else 'DINO_NOT_SUPPORTIVE_Z_LT_1')}

    out=Path('/tmp/dino_feature_bank.npz'); np.savez_compressed(out,embeddings=E.astype(np.float16),meta=z['meta'])
    feature_sha=hashlib.sha256(out.read_bytes()).hexdigest(); result['feature_bank_sha256']=feature_sha
    rp=Path('/tmp/dino_result.json'); rp.write_text(json.dumps(result,indent=2))
    api=HfApi(); api.upload_file(path_or_fileobj=str(out),path_in_repo='dino_feature_bank_vitb16.npz',repo_id=DATASET_REPO,repo_type='dataset',commit_message='Persist frozen DINOv3-B feature bank'); api.upload_file(path_or_fileobj=str(rp),path_in_repo='dino_result_vitb16.json',repo_id=DATASET_REPO,repo_type='dataset',commit_message='Persist DINOv3-B result')
    print('DINO_RESULT='+json.dumps(result,separators=(',',':')),flush=True)

if __name__=='__main__': main()
