#!/usr/bin/env python3
import os, io, json, time, base64, hashlib, math, random, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import torch
from PIL import Image, ImageOps, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings('ignore')
SEED = 20260803
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
RUN_ID = os.environ.get('RUN_ID', 'unknown')
MODEL_ID = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
MANIFEST = 'https://iiif.archive.org/iiif/bncf-pal.-586-images/manifest.json'
SUPABASE = 'https://ymaqlcfjmdwncdbjprmw.supabase.co'
ANON = 'sb_publishable_BOm91KbAPOZDCQ7H3yLFzw_VtNPk2ap'
UPLOAD_ENDPOINT = SUPABASE + '/functions/v1/p586-root-upload-v01'
PREFIX = 'p586_root_v01/dinov3_scan_' + RUN_ID
PAGE_INDICES = list(range(22, 80))
STORAGE_BASE = SUPABASE + '/storage/v1/object/public/manuscripts/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
S = requests.Session(); S.headers.update({'User-Agent':'VoynichRootResearch/0.1'})


def log(event, **kw):
    print(json.dumps({'event':event,'time':time.time(),**kw}, sort_keys=True), flush=True)

def get(url, tries=5, timeout=90, headers=None, params=None):
    last=None
    for k in range(tries):
        try:
            r=S.get(url,timeout=timeout,headers=headers,params=params)
            r.raise_for_status(); return r
        except Exception as e:
            last=e; time.sleep(min(15,1.5**k+random.random()))
    raise RuntimeError(f'GET failed {url}: {last}')

def upload(relpath, ctype, data):
    payload={'path':f'{PREFIX}/{relpath}','content_type':ctype,'data_b64':base64.b64encode(data).decode()}
    r=S.post(UPLOAD_ENDPOINT,headers={'x-upload-token':os.environ['UPLOAD_TOKEN'],'content-type':'application/json'},json=payload,timeout=180)
    r.raise_for_status(); out=r.json(); log('upload',path=out.get('path'),bytes=out.get('bytes')); return out

def parse_manifest(m):
    return m.get('items') or (m.get('sequences') or [{}])[0].get('canvases') or []

def image_url(c,w=1600):
    try:
        b=c['items'][0]['items'][0]['body']; sv=b.get('service')
        if isinstance(sv,list): sv=sv[0] if sv else None
        if isinstance(sv,dict):
            base=sv.get('id') or sv.get('@id')
            if base:return base.rstrip('/')+f'/full/{w},/0/default.jpg'
        return b.get('id') or b.get('@id')
    except Exception: pass
    try:
        b=c['images'][0]['resource']; sv=b.get('service')
        if isinstance(sv,list): sv=sv[0] if sv else None
        if isinstance(sv,dict):
            base=sv.get('@id') or sv.get('id')
            if base:return base.rstrip('/')+f'/full/{w},/0/default.jpg'
        return b.get('@id') or b.get('id')
    except Exception:return None

def load_image(url, tries=4):
    r=get(url,tries=tries,timeout=120)
    return ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert('RGB')

def fetch_refs():
    url=SUPABASE+'/rest/v1/herbal_objects'
    params={'select':'slug,obj_index,part,crop_path,crop_qa','manuscript_id':'eq.voynich','part':'in.(root,plant)','crop_path':'not.is.null','limit':'1000'}
    rows=get(url,headers={'apikey':ANON},params=params).json()
    out=[]
    for r in rows:
        qa=r.get('crop_qa')
        if qa in ('spurious_fragment','genuine_miss'): continue
        out.append(r)
    return out

def feature_batch(model, proc, images, batch=24):
    feats=[]
    for s in range(0,len(images),batch):
        xs=images[s:s+batch]
        inp=proc(images=xs,return_tensors='pt').to(DEVICE)
        with torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.float16,enabled=DEVICE=='cuda'):
            out=model(**inp)
            h=out.last_hidden_state.float()
            cls=h[:,0]
            patches=h[:,1:]
            f=torch.cat([cls,patches.mean(1),patches.amax(1)],dim=1)
            f=torch.nn.functional.normalize(f,dim=1)
        feats.append(f.cpu().numpy().astype('float32'))
        log('embed_progress',done=min(s+batch,len(images)),total=len(images))
    return np.concatenate(feats,axis=0)

def content_stats(im):
    a=np.asarray(im.convert('RGB').resize((128,128)),dtype=np.float32)
    border=np.concatenate([a[:8].reshape(-1,3),a[-8:].reshape(-1,3),a[:, :8].reshape(-1,3),a[:, -8:].reshape(-1,3)],axis=0)
    bg=np.median(border,axis=0)
    dist=np.linalg.norm(a-bg,axis=2)
    nonbg=float((dist>24).mean())
    g=np.asarray(im.convert('L').resize((128,128)),dtype=np.float32)/255
    gy,gx=np.gradient(g); edge=float((np.hypot(gx,gy)>.06).mean())
    return nonbg,edge,float(g.std())

def iou(a,b):
    x0=max(a[0],b[0]); y0=max(a[1],b[1]); x1=min(a[2],b[2]); y1=min(a[3],b[3])
    inter=max(0,x1-x0)*max(0,y1-y0)
    aa=max(1,(a[2]-a[0])*(a[3]-a[1])); bb=max(1,(b[2]-b[0])*(b[3]-b[1]))
    return inter/(aa+bb-inter)

def windows(im):
    W,H=im.size; out=[]
    scales=[(.34,.20),(.45,.25),(.58,.30)]
    xs=[.23,.50,.77]; ys=[.56,.67,.78,.88]
    for si,(wf,hf) in enumerate(scales):
        w=max(96,int(W*wf)); h=max(96,int(H*hf))
        for xi,cx in enumerate(xs):
            for yi,cy in enumerate(ys):
                x0=int(cx*W-w/2); y0=int(cy*H-h/2)
                x0=max(0,min(W-w,x0)); y0=max(0,min(H-h,y0))
                box=(x0,y0,x0+w,y0+h)
                crop=im.crop(box)
                nbg,edge,sd=content_stats(crop)
                out.append({'box':box,'scale':si,'xi':xi,'yi':yi,'crop':crop,'nonbg':nbg,'edge':edge,'sd':sd})
    return out

def make_crop_sheet(proposals, start, count=12):
    batch=proposals[start:start+count]
    cols=4; rows=math.ceil(len(batch)/cols); cw=300; ch=270
    sheet=Image.new('RGB',(cols*cw,rows*ch),'white'); d=ImageDraw.Draw(sheet)
    for j,p in enumerate(batch):
        x=(j%cols)*cw; y=(j//cols)*ch
        im=p['crop'].copy(); im.thumbnail((280,205))
        sheet.paste(im,(x+(cw-im.width)//2,y+35))
        d.rectangle((x,y,x+cw-1,y+ch-1),outline='black',width=1)
        d.text((x+6,y+5),f"p586 {p['page']:03d} rank {p['rank']} score {p['score']:.3f}",fill='black')
        d.text((x+6,y+20),f"rootP {p['root_prob']:.2f} nn {p['root_nn']:.3f} spec {p['specificity']:.3f}",fill='black')
    bio=io.BytesIO(); sheet.save(bio,'JPEG',quality=90,optimize=True); return bio.getvalue()

def make_page_sheet(page_records,start,count=6):
    batch=page_records[start:start+count]
    cols=3; rows=math.ceil(len(batch)/cols); cw=400; ch=560
    sheet=Image.new('RGB',(cols*cw,rows*ch),'white'); d=ImageDraw.Draw(sheet)
    for j,r in enumerate(batch):
        x=(j%cols)*cw; y=(j//cols)*ch
        im=r['page_image'].copy(); draw=ImageDraw.Draw(im)
        colors=['red','blue']
        for p in r['selected']:
            draw.rectangle(p['box'],outline=colors[p['rank']-1],width=max(3,im.width//300))
        im.thumbnail((370,500)); sheet.paste(im,(x+(cw-im.width)//2,y+40))
        d.rectangle((x,y,x+cw-1,y+ch-1),outline='black',width=1)
        d.text((x+8,y+8),f"canvas {r['page']:03d}",fill='black')
        d.text((x+8,y+23),f"red=rank1 blue=rank2",fill='black')
    bio=io.BytesIO(); sheet.save(bio,'JPEG',quality=88,optimize=True); return bio.getvalue()

protocol={
 'protocol_id':'P586-ROOT-DINOV3-V0.1-20260803','run_id':RUN_ID,'seed':SEED,'model_id':MODEL_ID,
 'manifest':MANIFEST,'pages':PAGE_INDICES,'window_scales':[[.34,.20],[.45,.25],[.58,.30]],
 'window_centres_x':[.23,.50,.77],'window_centres_y':[.56,.67,.78,.88],
 'feature':'concat(cls,mean_patch,max_patch), L2-normalized','selection':'two non-overlapping proposals per page',
 'purpose':'proposal generation only; no source/exemplar claim'
}
upload('protocol.json','application/json',json.dumps(protocol,indent=2,sort_keys=True).encode())
log('start',device=DEVICE,run_id=RUN_ID)

refs=fetch_refs(); log('refs_found',n=len(refs),roots=sum(r['part']=='root' for r in refs),plants=sum(r['part']=='plant' for r in refs))
ref_images=[]; ref_meta=[]
for k,r in enumerate(refs,1):
    try:
        im=load_image(STORAGE_BASE+r['crop_path'])
        ref_images.append(im); ref_meta.append(r)
    except Exception as e: log('ref_fail',path=r['crop_path'],error=str(e))
    if k%40==0: log('ref_download',done=k,total=len(refs))

proc=AutoImageProcessor.from_pretrained(MODEL_ID)
model=AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
refX=feature_batch(model,proc,ref_images)
y=np.array([1 if r['part']=='root' else 0 for r in ref_meta],dtype=int)
groups=np.array([r['slug'] for r in ref_meta])
cv=GroupKFold(n_splits=5)
clf=LogisticRegression(C=.1,max_iter=2000,class_weight='balanced',solver='liblinear',random_state=SEED)
auc_scores=cross_val_score(clf,refX,y,groups=groups,cv=cv,scoring='roc_auc')
clf.fit(refX,y)
log('calibration',auc_mean=float(auc_scores.mean()),auc_sd=float(auc_scores.std()),folds=auc_scores.tolist())
rootX=refX[y==1]; plantX=refX[y==0]

manifest=get(MANIFEST).json(); canvases=parse_manifest(manifest)
page_records=[]; all_candidates=[]
for pi in PAGE_INDICES:
    u=image_url(canvases[pi],1600)
    try: im=load_image(u)
    except Exception as e:
        log('page_fail',page=pi,error=str(e)); continue
    ws=windows(im)
    for w in ws:
        w.update({'page':pi,'page_url':u,'page_image':im})
    all_candidates.extend(ws); page_records.append({'page':pi,'page_url':u,'page_image':im,'candidates':ws})
    log('page_windows',page=pi,n=len(ws),size=im.size)

candX=feature_batch(model,proc,[c['crop'] for c in all_candidates])
root_sim=candX@rootX.T; plant_sim=candX@plantX.T
root_nn=np.sort(root_sim,axis=1)[:,-5:].mean(1)
plant_nn=np.sort(plant_sim,axis=1)[:,-5:].mean(1)
specificity=root_nn-plant_nn
root_prob=clf.predict_proba(candX)[:,1]
def z(v): return (v-v.mean())/(v.std()+1e-9)
score=.55*z(root_prob)+.35*z(root_nn)+.25*z(specificity)+.08*z(np.array([c['nonbg'] for c in all_candidates]))+.04*z(np.array([c['edge'] for c in all_candidates]))
for i,c in enumerate(all_candidates):
    c['feature']=candX[i]; c['root_nn']=float(root_nn[i]); c['plant_nn']=float(plant_nn[i]); c['specificity']=float(specificity[i]); c['root_prob']=float(root_prob[i]); c['score']=float(score[i])

proposals=[]
for r in page_records:
    ranked=sorted(r['candidates'],key=lambda c:c['score'],reverse=True)
    selected=[]
    for c in ranked:
        if c['nonbg']<.015 or c['sd']<.025: continue
        if all(iou(c['box'],q['box'])<.45 for q in selected):
            selected.append(c)
        if len(selected)==2: break
    if not selected: selected=ranked[:1]
    for rank,c in enumerate(selected,1):
        c['rank']=rank; proposals.append(c)
    r['selected']=selected

proposal_rows=[]
for p in proposals:
    bio=io.BytesIO(); p['crop'].save(bio,'PNG',optimize=True)
    rel=f"crops/canvas_{p['page']:03d}_rank{p['rank']}.png"
    up=upload(rel,'image/png',bio.getvalue())
    proposal_rows.append({
      'page':p['page'],'rank':p['rank'],'box':list(map(int,p['box'])),'page_url':p['page_url'],
      'crop_path':up['path'],'crop_url':up['public_url'],'root_prob':p['root_prob'],'root_nn':p['root_nn'],
      'plant_nn':p['plant_nn'],'specificity':p['specificity'],'score':p['score'],'nonbg':p['nonbg'],'edge':p['edge'],'sd':p['sd'],
      'window':{'scale':p['scale'],'xi':p['xi'],'yi':p['yi']},'qa_status':'unreviewed'
    })

for s in range(0,len(proposals),12): upload(f'contact_sheets/crops_{s:03d}-{min(s+11,len(proposals)-1):03d}.jpg','image/jpeg',make_crop_sheet(proposals,s,12))
for s in range(0,len(page_records),6): upload(f'contact_sheets/pages_{s:03d}-{min(s+5,len(page_records)-1):03d}.jpg','image/jpeg',make_page_sheet(page_records,s,6))

buf=io.BytesIO(); np.savez_compressed(buf,proposal_embeddings=np.stack([p['feature'] for p in proposals]),reference_embeddings=refX,reference_labels=y,proposal_pages=np.array([p['page'] for p in proposals]),proposal_ranks=np.array([p['rank'] for p in proposals]))
features_up=upload('features.npz','application/octet-stream',buf.getvalue())
report={
 'protocol':protocol,'device':DEVICE,'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
 'reference_counts':{'total':len(ref_meta),'roots':int(y.sum()),'plants':int((1-y).sum())},
 'calibration':{'grouped_auc_mean':float(auc_scores.mean()),'grouped_auc_sd':float(auc_scores.std()),'folds':[float(x) for x in auc_scores]},
 'pages_downloaded':len(page_records),'candidate_windows':len(all_candidates),'proposals':len(proposal_rows),
 'features':features_up,'proposal_rows':proposal_rows,
 'score_summary':{'mean':float(np.mean([p['score'] for p in proposals])),'sd':float(np.std([p['score'] for p in proposals])),'min':float(np.min([p['score'] for p in proposals])),'max':float(np.max([p['score'] for p in proposals]))}
}
blob=json.dumps(report,sort_keys=True,separators=(',',':')).encode(); report['result_sha256']=hashlib.sha256(blob).hexdigest()
upload('proposals.json','application/json',json.dumps(proposal_rows,indent=2,sort_keys=True).encode())
upload('run_report.json','application/json',json.dumps(report,indent=2,sort_keys=True).encode())
print('RESULT_JSON='+json.dumps({'run_id':RUN_ID,'prefix':PREFIX,'pages':len(page_records),'windows':len(all_candidates),'proposals':len(proposal_rows),'auc':report['calibration'],'sha256':report['result_sha256']},sort_keys=True),flush=True)
