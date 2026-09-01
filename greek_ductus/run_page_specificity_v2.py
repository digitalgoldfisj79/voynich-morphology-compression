#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,re
from pathlib import Path
from collections import Counter
import numpy as np

import extract_cpu as ec
import run_specificity_cpu_v1 as base

ROOT=Path(__file__).parent
SPEC=json.loads((ROOT/'page_specificity_manifest_v2.json').read_text())
V1=base.SPEC
MODEL='facebook/dinov3-vitb16-pretrain-lvd1689m'
REV='5931719e67bbdb9737e363e781fb0c67687896bc'


def norm(v):
    v=np.asarray(v,np.float32); return v/(np.linalg.norm(v)+1e-12)
def cosdist(a,b): return float(1.0-np.dot(norm(a),norm(b)))
def euclid(a,b): return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))

def label_text(x):
    if isinstance(x,dict):
        vals=[]
        for v in x.values(): vals.extend(v if isinstance(v,list) else [v])
        x=' '.join(map(str,vals))
    return str(x or '')

def normfolio(x):
    s=label_text(x).lower().strip()
    s=re.sub(r'\s*\([^)]*\)\s*','',s)
    s=s.replace('folio','').replace('fol.','').replace('f.','').strip()
    if s.startswith('f'): s=s[1:]
    m=re.search(r'0*(\d+)\s*([rv])\b',s)
    if m:return str(int(m.group(1)))+m.group(2)
    return s

def canonical_image_urls(c):
    """Acquisition-only fallbacks. Scientific target/page selection is unchanged."""
    urls=[]
    try:
        r=c['images'][0]['resource']; direct=r.get('@id') or r.get('id'); svc=r.get('service') or {}; sid=svc.get('@id') or svc.get('id')
        if sid:
            urls.extend([sid.rstrip('/')+'/full/1600,/0/default.jpg',sid.rstrip('/')+'/full/full/0/default.jpg'])
        if direct: urls.append(direct)
    except Exception:
        pass
    try:
        u=ec.image_url(c)
        if u: urls.insert(0,u)
    except Exception:
        pass
    out=[]
    for u in urls:
        if u and u not in out: out.append(u)
    return out

def fetch_canvas_image(c,oid,lab):
    last=None
    for u in canonical_image_urls(c):
        try:
            return base.fetch_img(u)
        except Exception as e:
            last=e; print(json.dumps({'event':'image_fallback','object':oid,'folio':lab,'url':u,'error':repr(e)}),flush=True)
    raise last or RuntimeError(f'{oid} {lab}: no image URL')

def exact_iiif_pages(manifest,oid,labels):
    cs=ec.canvases(base.get_json(manifest)); want=[normfolio(x) for x in labels]; found={}
    for i,c in enumerate(cs):
        nl=normfolio(c.get('label'))
        if nl in want and nl not in found: found[nl]=(i,c,label_text(c.get('label')))
    missing=[x for x in want if x not in found]
    if missing: raise RuntimeError(f'{oid}: missing labels {missing}; found keys sample={list(found)}')
    out=[]
    for lab in want:
        i,c,raw=found[lab]; im=fetch_canvas_image(c,oid,lab); cc=base.page_crops_from_image(im,oid,i)
        for _,m in cc: m['folio']=lab; m['raw_label']=raw
        out.extend(cc); print(json.dumps({'event':'exact_page','object':oid,'folio':lab,'index':i,'crops':len(cc)}),flush=True)
    return out

def collect():
    objects={}; meta={}
    # frozen ordinary Greek/Latin controls from replication1/specificity v1
    for fam,rows in V1['ordinary'].items():
        for r in rows:
            oid=fam+'::'+r['shelfmark']
            items=base.even_take(base.iiif_object(r['manifest'],oid,V1['control_page_fractions']),90)
            objects[oid]=items; meta[oid]={'family':fam,'type':'ordinary','status':'CONTROL','n':len(items)}
            print(json.dumps({'event':'object','object':oid,'n':len(items)}),flush=True)
    # exact executable page loci only. Unresolved hidden links remain in manifest but do not enter this run.
    for t in SPEC['verified_targets']:
        if t['source_type']=='precomputed_exact_region': continue
        oid=t['id']; st=t['source_type']
        if st=='iiif_labels': items=exact_iiif_pages(t['manifest'],oid,t['target_labels'])
        elif st=='direct_openn_index':
            im=base.fetch_img(t['target_url']); items=base.page_crops_from_image(im,oid,218)
            for _,m in items:m['folio']='PDF218'
        else: raise ValueError(st)
        items=base.even_take(items,180); objects[oid]=items
        meta[oid]={'family':'SPECIAL','type':'target','status':t['target_status'],'kind':t['target_kind'],'manuscript':t['manuscript'],'n':len(items)}
        print(json.dumps({'event':'target','object':oid,'n':len(items),'status':t['target_status']}),flush=True)
        if t.get('control_labels'):
            cid=oid+'::SAME_CODEX_ORDINARY'; cc=base.even_take(exact_iiif_pages(t['manifest'],cid,t['control_labels']),180)
            objects[cid]=cc; meta[cid]={'family':'INTERNAL','type':'internal_control','parent':oid,'n':len(cc)}
            print(json.dumps({'event':'internal_control','object':cid,'n':len(cc)}),flush=True)
    # same frozen VMS sample as specificity v1
    vi=base.even_take(base.iiif_object(V1['vms_manifest'],'VOYNICH',V1['vms_fractions']),270)
    objects['VOYNICH']=vi; meta['VOYNICH']={'family':'VOYNICH','type':'target_vms','n':len(vi)}
    return objects,meta

def cpu_run(objects,meta):
    raw={k:base.vec(v) for k,v in objects.items()}
    ordkeys=[k for k in raw if meta[k]['type']=='ordinary']
    fit=np.concatenate([raw[k] for k in ordkeys],axis=0)
    med,scale=base.robust_scale(fit); Z={k:(X-med)/scale for k,X in raw.items()}; ov={k:np.median(X,0) for k,X in Z.items()}
    vv=ov['VOYNICH']; cg=np.mean([ov[k] for k in ordkeys if meta[k]['family']=='GREEK'],0); cl=np.mean([ov[k] for k in ordkeys if meta[k]['family']=='LATIN'],0)
    result_core(vv,cg,cl,ov,meta,euclid,'cpu_stroke_geometry_page_v2')

def dino_run(objects,meta):
    import torch
    from transformers import AutoImageProcessor,AutoModel
    token=os.getenv('HF_TOKEN'); proc=AutoImageProcessor.from_pretrained(MODEL,revision=REV,token=token)
    model=AutoModel.from_pretrained(MODEL,revision=REV,token=token,torch_dtype=torch.float16,low_cpu_mem_usage=True).eval().cuda()
    E={}; total=sum(len(x) for x in objects.values()); done=0
    with torch.inference_mode():
        for oid,items in objects.items():
            ims=[im for im,_ in items]; chunks=[]
            for i in range(0,len(ims),64):
                x=proc(images=ims[i:i+64],return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda',dtype=torch.float16): y=model(**x).last_hidden_state[:,0].float()
                y=torch.nn.functional.normalize(y,dim=1); chunks.append(y.cpu().numpy()); done+=len(y)
                print(json.dumps({'event':'embed','object':oid,'done':done,'total':total}),flush=True)
            E[oid]=np.concatenate(chunks).astype(np.float32)
    ov={k:norm(v.mean(0)) for k,v in E.items()}; ordkeys=[k for k in ov if meta[k]['type']=='ordinary']
    vv=ov['VOYNICH']; cg=norm(np.mean([ov[k] for k in ordkeys if meta[k]['family']=='GREEK'],0)); cl=norm(np.mean([ov[k] for k in ordkeys if meta[k]['family']=='LATIN'],0))
    result_core(vv,cg,cl,ov,meta,cosdist,'dinov3_b_cls_page_v2',model_info={'model':MODEL,'revision':REV,'n_embeddings':total})

def result_core(vv,cg,cl,ov,meta,dist,representation,model_info=None):
    dG=dist(vv,cg); dL=dist(vv,cl)
    tids=[k for k in ov if meta[k]['type']=='target']
    targets={k:dist(vv,ov[k]) for k in tids}
    internal={}
    for k in tids:
        cid=k+'::SAME_CODEX_ORDINARY'
        if cid in ov:
            dc=dist(vv,ov[cid]); dt=targets[k]
            internal[k]={'d_target':dt,'d_same_codex_ordinary':dc,'ordinary_minus_target':dc-dt,'target_closer_than_same_codex_ordinary':bool(dt<dc)}
    rows=[]
    for k in tids:
        rows.append({'id':k,'status':meta[k]['status'],'kind':meta[k]['kind'],'manuscript':meta[k]['manuscript'],'distance_to_voynich':targets[k],'special_minus_greek':targets[k]-dG,'closer_than_ordinary_greek':bool(targets[k]<dG),'internal':internal.get(k)})
    rows.sort(key=lambda x:x['distance_to_voynich'])
    # ordinary-control classifier sanity check
    ordkeys=[k for k in ov if meta[k]['type']=='ordinary']; hits=0; conf=Counter()
    for k in ordkeys:
        fam=meta[k]['family']; gg=[ov[j] for j in ordkeys if j!=k and meta[j]['family']=='GREEK']; ll=[ov[j] for j in ordkeys if j!=k and meta[j]['family']=='LATIN']
        g=np.mean(gg,0); l=np.mean(ll,0)
        if representation.startswith('dinov3'):g,l=norm(g),norm(l)
        pred='GREEK' if dist(ov[k],g)<dist(ov[k],l) else 'LATIN'; hits+=pred==fam; conf[(fam,pred)]+=1
    precomputed={
      'GR201_F176R':{
        'source':'GR201_CIPHERLINE_RESULT_V1_20260901.json',
        'representation':'dinov3_b_cls_within_gr201',
        'd_vms_cipher_primary_bottom2':0.03593453620295106,
        'd_vms_same_page_plain':0.016032278424530144,
        'ordinary_minus_cipher':-0.019902257778420918,
        'decision':'NO_CIPHER_SPECIFIC_ENRICHMENT'
      }
    } if representation.startswith('dinov3') else {}
    result={'protocol':SPEC['protocol'],'representation':representation,'ordinary_baseline':{'d_vms_greek':dG,'d_vms_latin':dL,'greek_advantage_vs_latin':dL-dG,'loo_accuracy':hits/len(ordkeys),'loo_confusion':{f'{a}->{b}':n for (a,b),n in conf.items()}},'page_targets_ranked':rows,'same_codex_tests':internal,'precomputed_exact_regions':precomputed,'unresolved_archive_loci_excluded':SPEC['archive_loci_not_scored_until_link_resolved'],'nearest_executable_target':rows[0]['id'] if rows else None,'nearest_executable_target_distance':rows[0]['distance_to_voynich'] if rows else None,'nearest_executable_target_vs_greek_margin':(rows[0]['distance_to_voynich']-dG) if rows else None}
    if model_info:result.update(model_info)
    tag='PAGE_SPECIFICITY_DINO_V2=' if representation.startswith('dinov3') else 'PAGE_SPECIFICITY_CPU_V2='
    print(tag+json.dumps(result,separators=(',',':')),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['cpu','dino'],required=True); a=ap.parse_args()
    objects,meta=collect()
    cpu_run(objects,meta) if a.mode=='cpu' else dino_run(objects,meta)
