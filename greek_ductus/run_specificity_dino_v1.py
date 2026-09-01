#!/usr/bin/env python3
from __future__ import annotations
import os,json
from collections import Counter,defaultdict
import numpy as np, torch
from transformers import AutoImageProcessor,AutoModel
import run_specificity_cpu_v1 as cpu

SPEC=cpu.SPEC
MODEL='facebook/dinov3-vitb16-pretrain-lvd1689m'
REV='5931719e67bbdb9737e363e781fb0c67687896bc'
SEED=413
rng=np.random.default_rng(SEED)

def norm(v):
    v=np.asarray(v,np.float32); return v/(np.linalg.norm(v)+1e-12)
def cosdist(a,b): return float(1.0-np.dot(norm(a),norm(b)))

def collect():
    objects={}; meta={}; cropmeta={}
    for fam,rows in SPEC['ordinary'].items():
        for r in rows:
            oid=fam+'::'+r['shelfmark']; items=cpu.even_take(cpu.iiif_object(r['manifest'],oid,SPEC['control_page_fractions']),90)
            objects[oid]=[im for im,m in items]; cropmeta[oid]=[m for im,m in items]; meta[oid]={'family':fam,'type':'ordinary','n':len(items)}
            print(json.dumps({'event':'object','object':oid,'selected':len(items)}),flush=True)
    for s in SPEC['special']:
        oid=s['object']; st=s['source_type']
        if st=='iiif': items=cpu.iiif_object(s['manifest'],oid,SPEC['special_page_fractions'])
        elif st=='iiif_last6': items=cpu.iiif_object(s['manifest'],oid,last6=True)
        elif st=='openn': items=cpu.openn_object(s['html'],oid,SPEC['special_page_fractions'])
        else: raise ValueError(st)
        items=cpu.even_take(items,180); objects[oid]=[im for im,m in items]; cropmeta[oid]=[m for im,m in items]; meta[oid]={'family':s['family'],'type':'special','n':len(items),'note':s.get('note')}
        print(json.dumps({'event':'object','object':oid,'selected':len(items)}),flush=True)
    items=cpu.even_take(cpu.iiif_object(SPEC['vms_manifest'],'VOYNICH',SPEC['vms_fractions']),270)
    objects['VOYNICH']=[im for im,m in items]; cropmeta['VOYNICH']=[m for im,m in items]; meta['VOYNICH']={'family':'VOYNICH','type':'target','n':len(items)}
    return objects,meta,cropmeta

def embed_all(objects):
    token=os.getenv('HF_TOKEN'); proc=AutoImageProcessor.from_pretrained(MODEL,revision=REV,token=token)
    model=AutoModel.from_pretrained(MODEL,revision=REV,token=token,torch_dtype=torch.float16,low_cpu_mem_usage=True).eval().cuda()
    out={}; done=0; total=sum(len(v) for v in objects.values())
    with torch.inference_mode():
        for oid,ims in objects.items():
            chunks=[]
            for i in range(0,len(ims),64):
                x=proc(images=ims[i:i+64],return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda',dtype=torch.float16): y=model(**x).last_hidden_state[:,0].float()
                y=torch.nn.functional.normalize(y,dim=1); chunks.append(y.cpu().numpy()); done+=len(y)
                print(json.dumps({'event':'dino_batch','object':oid,'done':done,'total':total}),flush=True)
            out[oid]=np.concatenate(chunks).astype(np.float32)
    return out

def main():
    objects,meta,cropmeta=collect(); E=embed_all(objects)
    ov={k:norm(v.mean(0)) for k,v in E.items()}; ordkeys=[k for k in ov if meta[k]['type']=='ordinary']
    cg=norm(np.mean([ov[k] for k in ordkeys if meta[k]['family']=='GREEK'],0)); cl=norm(np.mean([ov[k] for k in ordkeys if meta[k]['family']=='LATIN'],0)); vv=ov['VOYNICH']
    dG,dL=cosdist(vv,cg),cosdist(vv,cl)
    special={k:cosdist(vv,ov[k]) for k in ov if meta[k]['type']=='special'}; ns=min(special,key=special.get); margin=special[ns]-dG
    sf={}
    for fam in sorted({meta[k]['family'] for k in ov if meta[k]['type']=='special'}):
        ks=[k for k in ov if meta[k]['family']==fam]; sf[fam]=cosdist(vv,norm(np.mean([ov[k] for k in ks],0)))
    conf=Counter(); hits=0
    for k in ordkeys:
        fam=meta[k]['family']; gg=[ov[j] for j in ordkeys if j!=k and meta[j]['family']=='GREEK']; ll=[ov[j] for j in ordkeys if j!=k and meta[j]['family']=='LATIN']
        pred='GREEK' if cosdist(ov[k],norm(np.mean(gg,0)))<cosdist(ov[k],norm(np.mean(ll,0))) else 'LATIN'; conf[(fam,pred)]+=1; hits+=pred==fam
    labs=np.array([meta[k]['family'] for k in ordkeys]); X=np.stack([ov[k] for k in ordkeys]); obs=dL-dG; null=[]
    for _ in range(20000):
        p=rng.permutation(labs); a=norm(X[p=='GREEK'].mean(0)); b=norm(X[p=='LATIN'].mean(0)); null.append(cosdist(vv,b)-cosdist(vv,a))
    null=np.array(null); z=float((obs-null.mean())/(null.std(ddof=1)+1e-12)); ep=float((1+np.sum(null>=obs))/(len(null)+1))
    # VMS page readout, using exact selected crops
    by=defaultdict(list)
    for e,m in zip(E['VOYNICH'],cropmeta['VOYNICH']): by[m['page']].append(e)
    pages=[]
    for p,arr in sorted(by.items()):
        v=norm(np.mean(arr,0)); pages.append({'page':int(p),'n':len(arr),'d_G':cosdist(v,cg),'d_L':cosdist(v,cl),'greek_advantage_vs_latin':cosdist(v,cl)-cosdist(v,cg),'nearest_special':min(special,key=special.get)})
    result={'protocol':SPEC['protocol'],'representation':'dinov3_b_cls_specificity_v1','model':MODEL,'revision':REV,'embedding_dim':int(vv.size),'n_embeddings':sum(len(x) for x in E.values()),'objects':{k:{**meta[k],'distance_to_voynich':(0 if k=='VOYNICH' else cosdist(vv,ov[k]))} for k in ov},'distances':{'GREEK':dG,'LATIN':dL,'special_objects':special,'special_families':sf},'nearest_special':ns,'specificity_margin_special_minus_greek':margin,'nearest_overall':min({'GREEK':dG,'LATIN':dL,**special},key={'GREEK':dG,'LATIN':dL,**special}.get),'ordinary_loo_accuracy':hits/len(ordkeys),'ordinary_loo_confusion':{f'{a}->{b}':n for (a,b),n in conf.items()},'greek_vs_latin_reproduction':{'advantage':obs,'perm_mean':float(null.mean()),'perm_sd':float(null.std(ddof=1)),'z':z,'empirical_p_one_sided':ep,'permutations':20000},'vms_pages':pages,'decision_dino':('GREEK_SURVIVES_SPECIALS' if margin>0 else 'SPECIAL_ABSORBS_GREEK_SIGNAL')}
    print('SPECIFICITY_DINO_RESULT='+json.dumps(result,separators=(',',':')),flush=True)
if __name__=='__main__':main()
