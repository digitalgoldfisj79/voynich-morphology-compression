#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, itertools, json, math, os, time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import requests
from PIL import Image
from skimage.morphology import skeletonize

ROOT=Path(__file__).parent
UA={'User-Agent':'voynich-greek-affinity-fixedpatch/2026-09-01'}
STRICT_INK=(0.03,0.30)
BASE_INK=(0.02,0.35)
MAX_MS=360
MAX_VMS=720

HANDS={
'Hand1':['f1r','f1v','f2r','f2v','f3r','f3v','f4r','f4v','f5r','f5v','f6r','f6v','f7r','f7v','f8r','f8v','f9r','f9v','f10r','f10v','f11r','f11v','f13r','f13v','f14r','f14v','f15r','f15v','f16r','f16v','f17r','f17v','f18r','f18v','f19r','f19v','f20r','f20v','f21r','f21v','f22r','f22v','f23r','f23v','f24r','f24v','f25r','f25v','f27r','f27v','f28r','f28v','f29r','f29v','f30r','f30v','f32r','f32v','f35r','f35v','f36r','f36v','f37r','f37v','f38r','f38v','f42r','f42v','f44r','f44v','f45r','f45v','f47r','f47v','f49r','f49v','f51r','f51v','f52r','f52v','f53r','f53v','f54r','f54v','f56r','f56v','f57v','f87r','f87v','f88r','f88v','f89r1','f89r2','f89v2','f89v1','f90r1','f90r2','f90v2','f90v1','f93r','f93v','f96r','f96v','f99r','f99v','f100r','f100v','f101r','f101v','f102r1','f102r2','f102v2','f102v1'],
'Hand2':['f26r','f26v','f31r','f31v','f33r','f33v','f34r','f34v','f39r','f39v','f40r','f40v','f43r','f43v','f46r','f46v','f50r','f50v','f55r','f55v','f75r','f75v','f76r','f76v','f77r','f77v','f78r','f78v','f79r','f79v','f80r','f80v','f81r','f81v','f82r','f82v','f83r','f83v','f84r','f84v','f85r1','f85r2','f86v4','f86v6','f86v5','f86v3'],
'Hand3':['f58r','f58v','f65r','f65v','f94r','f94v','f95r1','f95r2','f95v2','f95v1','f103r','f103v','f104r','f104v','f105r','f105v','f106r','f106v','f107r','f107v','f108r','f108v','f111r','f111v','f112r','f112v','f113r','f113v','f114r','f114v','f115v','f116r'], # mixed f115r excluded
'Hand4':['f67r1','f67r2','f67v2','f67v1','f68r1','f68r2','f68r3','f68v3','f68v2','f68v1','f69r','f69v','f70r1','f70r2','f70v2','f70v1','f71r','f71v','f72r1','f72r2','f72r3','f72v3','f72v2','f72v1','f73r','f73v','fRos'],
'Hand5':['f41r','f41v','f48r','f48v','f57r','f66r','f66v']}


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

def canvases(m):
    out=[]
    if 'items' in m:
        for i,c in enumerate(m.get('items',[])):
            try:
                body=c['items'][0]['items'][0]['body']; srv=body.get('service') or []
                if isinstance(srv,dict): srv=[srv]
                sid=(srv[0].get('id') or srv[0].get('@id')) if srv else None
                out.append({'i':i,'service':sid,'body':body.get('id') or body.get('@id'),'label':c.get('label')})
            except Exception: pass
    else:
        for i,c in enumerate((m.get('sequences') or [{}])[0].get('canvases',[])):
            try:
                res=c['images'][0]['resource']; srv=res.get('service') or {}
                if isinstance(srv,list): srv=srv[0] if srv else {}
                out.append({'i':i,'service':srv.get('@id') or srv.get('id'),'body':res.get('@id') or res.get('id'),'label':c.get('label')})
            except Exception: pass
    return [x for x in out if x.get('service') or x.get('body')]
def image_url(c,width=1600):
    sid=c.get('service'); body=c.get('body')
    if sid:
        if 'images.iiif.slub-dresden.de' in sid and body: return body
        return sid.rstrip('/')+f'/full/{width},/0/default.jpg'
    if body and '/full/' in body: return body.split('/full/')[0]+f'/full/{width},/0/default.jpg'
    return body

def label_text(x):
    if isinstance(x,str): return x
    if isinstance(x,dict):
        vals=[]
        for v in x.values(): vals.extend(v if isinstance(v,list) else [v])
        return ' '.join(map(str,vals))
    return str(x or '')
def norm_label(s):
    s=label_text(s).lower().replace(' ','').replace('.','')
    for pre in ['folio','fol','f.']: s=s.replace(pre,'')
    return s

def ink_mask(im):
    a=np.array(im.convert('L')); h,w=a.shape; a=a[int(.035*h):int(.965*h),int(.035*w):int(.965*w)]
    a=cv2.GaussianBlur(a,(3,3),0); block=max(31,((min(a.shape)//35)//2)*2+1); block=min(block,101); block=block if block%2 else block+1
    bw=cv2.adaptiveThreshold(a,255,cv2.ADAPTIVE_THRESH_GAUSSI_C if hasattr(cv2,'ADAPTIVE_THRESH_GAUSSI_C') else cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,17)
    return cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))

def smooth1d(x,k=5): return np.convolve(x,np.ones(k)/k,mode='same')
def line_bands(bw):
    H,W=bw.shape; row=(bw>0).sum(1).astype(float); sm=smooth1d(row,5); p95=float(np.percentile(sm,95)); thr=max(3.0,0.08*p95); active=sm>thr
    runs=[]; st=None
    for i,v in enumerate(active):
        if v and st is None: st=i
        if st is not None and ((not v) or i==H-1):
            en=i if not v else i+1; runs.append([st,en]); st=None
    merged=[]
    for r in runs:
        if merged and r[0]-merged[-1][1]<=3: merged[-1][1]=r[1]
        else: merged.append(r)
    out=[]
    for y0,y1 in merged:
        hh=y1-y0
        if hh<8 or hh>0.08*H: continue
        sub=bw[y0:y1]; cov=float(np.mean(np.any(sub>0,axis=0)))
        if cov<0.08: continue
        pad=max(2,int(round(.20*hh))); out.append((max(0,y0-pad),min(H,y1+pad),hh/H))
    return out

def fixed_patches(bw,ink_gate=BASE_INK):
    H,W=bw.shape; out=[]
    for li,(y0,y1,relh) in enumerate(line_bands(bw)):
        lh=max(8,y1-y0); ww=int(np.clip(4*lh,96,256)); stride=max(1,ww//2)
        xs=list(range(0,max(1,W-ww+1),stride))
        if xs and xs[-1] != W-ww and W>ww: xs.append(W-ww)
        for x0 in xs:
            p=bw[y0:y1,x0:x0+ww]; dens=float(np.mean(p>0))
            if not (ink_gate[0] <= dens <= ink_gate[1]): continue
            n,lab,st,_=cv2.connectedComponentsWithStats((p>0).astype(np.uint8),8); cc=n-1
            if not (2<=cc<=60): continue
            out.append((p.copy(),{'line':li,'x':x0,'line_rel_height':relh,'ink_fraction':dens,'components':cc}))
    return out

def entropy(v):
    v=np.asarray(v,float); s=v.sum()
    if s<=0: return 0.0
    q=v/s; q=q[q>0]; return float(-(q*np.log2(q)).sum())
def skeleton_features(p):
    fg=p>0; sk=skeletonize(fg); H,W=sk.shape; coords=np.argwhere(sk); L=max(1,len(coords)); area=H*W
    ep=jn=0; orient=[]; turn=[]
    nbrs=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for y,x in coords:
        ns=[]
        for dy,dx in nbrs:
            yy,xx=y+dy,x+dx
            if 0<=yy<H and 0<=xx<W and sk[yy,xx]: ns.append((dy,dx))
        deg=len(ns)
        if deg==1: ep+=1
        elif deg>=3: jn+=1
        for dy,dx in ns:
            if (dy>0) or (dy==0 and dx>0): orient.append(math.atan2(dy,dx)%math.pi)
        if deg==2:
            a=np.array(ns[0],float); b=np.array(ns[1],float); a/=np.linalg.norm(a); b/=np.linalg.norm(b)
            c=float(np.clip(np.dot(-a,b),-1,1)); turn.append(math.acos(c))
    oh,_=np.histogram(orient,bins=12,range=(0,math.pi)); th,_=np.histogram(turn,bins=12,range=(0,math.pi))
    oh=oh/max(1,oh.sum()); th=th/max(1,th.sum())
    n,lab,st,_=cv2.connectedComponentsWithStats(fg.astype(np.uint8),8); cc=max(1,n-1)
    contours,_=cv2.findContours((fg.astype(np.uint8)*255),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    # Euler holes from contour hierarchy: use connected-components on inverted interior approximation
    holes=0
    cnts,hier=cv2.findContours((fg.astype(np.uint8)*255),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None: holes=sum(1 for h in hier[0] if h[3]>=0)
    morph=np.r_[len(coords)/area, ep/L, jn/L, holes/cc, oh, th].astype(float)
    areas=st[1:,cv2.CC_STAT_AREA].astype(float)/area if n>1 else np.array([0.])
    row=(fg).sum(1); col=(fg).sum(0)
    gross=np.array([fg.mean(),cc,np.median(areas),entropy(row),entropy(col)],float)
    return morph,gross

def even_take(items,n):
    if len(items)<=n: return items
    ix=np.linspace(0,len(items)-1,n).round().astype(int); return [items[int(i)] for i in ix]
def agg(X):
    X=np.asarray(X,float); return np.r_[np.median(X,axis=0),np.percentile(X,75,axis=0)-np.percentile(X,25,axis=0)]
def robust_scale(M,q):
    med=np.median(M,axis=0); mad=np.median(np.abs(M-med),axis=0)*1.4826; tiny=mad<1e-9; mad[tiny]=1.0
    return (M-med)/mad,(q-med)/mad,int(tiny.sum())
def dist(a,b): return float(np.linalg.norm(a-b)/math.sqrt(len(a)))
def medcent(M): return np.median(M,axis=0)
def exact_test(M,labs,q):
    M,q,tiny=robust_scale(M,q); labs=np.asarray(labs,dtype=object); cg=medcent(M[labs=='GREEK']); cl=medcent(M[labs=='LATIN']); dg=dist(q,cg); dl=dist(q,cl); A=dl-dg
    vals=[]; n=len(labs); idx=np.arange(n)
    for gix in itertools.combinations(range(n),n//2):
        g=np.zeros(n,bool); g[list(gix)]=True
        if np.array_equal(g,labs=='GREEK'): continue
        vals.append(dist(q,medcent(M[~g]))-dist(q,medcent(M[g])))
    vals=np.asarray(vals,float); mu=float(vals.mean()); sd=float(vals.std(ddof=1)); z=(A-mu)/sd; p=float((1+np.sum(vals>=A))/(len(vals)+1))
    loo=[]; good=0
    for i in range(n):
        keep=idx!=i; lm=labs[keep]; mm=M[keep]; d={'GREEK':dist(M[i],medcent(mm[lm=='GREEK'])),'LATIN':dist(M[i],medcent(mm[lm=='LATIN']))}; pred=min(d,key=d.get); good+=pred==labs[i]; loo.append({'i':i,'true':labs[i],'pred':pred,'distances':d})
    return {'distances':{'GREEK':dg,'LATIN':dl},'greek_advantage':A,'null_mean':mu,'null_sd':sd,'z':z,'exact_p_one_sided':p,'null_assignments':len(vals),'tiny_mad_dims':tiny,'loo_accuracy':good/n,'loo':loo},(cg,cl),(M,q)

def get_pages(manifest,fracs):
    cs=canvases(retry_json(manifest,'manifest')); inds=sorted(set(max(0,min(len(cs)-1,int(round(f*(len(cs)-1))))) for f in fracs)); return cs,inds

def acquire_page(c,label,ink_gate):
    im=retry_image(image_url(c),label); bw=ink_mask(im); ps=fixed_patches(bw,ink_gate); out=[]
    for p,m in ps:
        mf,gf=skeleton_features(p); gf=np.r_[gf,m['line_rel_height']]
        out.append((p,m,mf,gf))
    return out

def build_bank(spec,ink_gate=BASE_INK,include_hands=True):
    bank=[]; page_diag=[]
    for fam in ['GREEK','LATIN']:
        for r in spec['families'][fam]:
            cs,inds=get_pages(r['manifest'],spec['control_page_fractions']); items=[]; yields=[]
            for pi in inds:
                got=acquire_page(cs[pi],f'{fam}:{r["shelfmark"]}:{pi}',ink_gate); yields.append(len(got));
                for z in got: items.append((pi,)+z)
                print(json.dumps({'event':'fixed_page','family':fam,'shelfmark':r['shelfmark'],'page':pi,'patches':len(got)}),flush=True)
            chosen=even_take(items,MAX_MS); bank.append({'family':fam,'id':r['shelfmark'],'items':chosen,'page_yields':yields}); print(json.dumps({'event':'fixed_ms','family':fam,'shelfmark':r['shelfmark'],'available':len(items),'selected':len(chosen)}),flush=True)
    vcs,vinds=get_pages(spec['vms_manifest'],spec['vms_fractions']); items=[]; yields=[]
    for pi in vinds:
        got=acquire_page(vcs[pi],f'VMS:{pi}',ink_gate); yields.append(len(got));
        for z in got: items.append((pi,)+z)
        print(json.dumps({'event':'fixed_page','family':'VOYNICH','page':pi,'label':label_text(vcs[pi].get('label')),'patches':len(got)}),flush=True)
    chosen=even_take(items,MAX_VMS); bank.append({'family':'VOYNICH','id':'Beinecke408','items':chosen,'page_yields':yields})
    hand_bank={}
    if include_hands:
        labelmap={norm_label(c.get('label')):c for c in vcs}
        for hand,folios in HANDS.items():
            hitems=[]; hy=[]; missing=[]
            for fol in folios:
                c=labelmap.get(norm_label(fol))
                if c is None: missing.append(fol); continue
                got=acquire_page(c,f'{hand}:{fol}',ink_gate); hy.append(len(got))
                for z in got: hitems.append((fol,)+z)
            hitems=even_take(hitems,MAX_MS); hand_bank[hand]={'items':hitems,'page_yields':hy,'missing':missing,'requested':len(folios)}
            print(json.dumps({'event':'hand_bank','hand':hand,'pages_found':len(folios)-len(missing),'missing':len(missing),'patches':len(hitems)}),flush=True)
    return bank,hand_bank

def summaries(bank):
    ms=[]; v=None
    for r in bank:
        morph=[x[-2] for x in r['items']]; gross=[x[-1] for x in r['items']]
        # add yield summary only to gross manuscript representation
        gs=agg(gross); y=np.array(r['page_yields'],float)[:,None] if r['page_yields'] else np.zeros((1,1)); gs=np.r_[gs,agg(y)]
        rec={'family':r['family'],'id':r['id'],'n':len(morph),'morph':agg(morph),'gross':gs}
        if r['family']=='VOYNICH': v=rec
        else: ms.append(rec)
    return ms,v

def hand_results(hand_bank,ctrl_ms,ctrl_labs,ctrl_morph):
    # scale and centroids frozen from controls only
    med=np.median(ctrl_morph,axis=0); mad=np.median(np.abs(ctrl_morph-med),axis=0)*1.4826; mad[mad<1e-9]=1.0; M=(ctrl_morph-med)/mad; cg=medcent(M[ctrl_labs=='GREEK']); cl=medcent(M[ctrl_labs=='LATIN'])
    out=[]
    for h,r in hand_bank.items():
        if len(r['items'])<10: out.append({'hand':h,'status':'LOW_YIELD','n':len(r['items']),'missing':r['missing']}); continue
        q=(agg([x[-2] for x in r['items']])-med)/mad; dg=dist(q,cg); dl=dist(q,cl); out.append({'hand':h,'n':len(r['items']),'pages_found':r['requested']-len(r['missing']),'missing':r['missing'],'distances':{'GREEK':dg,'LATIN':dl},'greek_advantage':dl-dg})
    return out

def cpu_mode(ink_gate,include_hands=True):
    spec=json.loads((ROOT/'replication1_manifest.json').read_text()); bank,hb=build_bank(spec,ink_gate,include_hands); ms,v=summaries(bank); labs=np.array([r['family'] for r in ms],dtype=object); M=np.stack([r['morph'] for r in ms]); G=np.stack([r['gross'] for r in ms])
    head,_,_=exact_test(M,labs,v['morph']); gross,_,_=exact_test(G,labs,v['gross']); hands=hand_results(hb,ms,labs,M) if include_hands else []
    result={'protocol':'2026-09-01.greek-affinity.final-fixedpatch.v1','mode':'cpu','ink_gate':list(ink_gate),'headline':head,'gross_null':gross,'per_manuscript':[{'family':r['family'],'id':r['id'],'n':r['n']} for r in ms],'vms_n':v['n'],'hands':hands,'decision':('PASS_TO_DINO' if head['z']>=2 and head['greek_advantage']>0 and gross['z']<2 else 'DO_NOT_PROMOTE')}
    print('FINAL_FIXEDPATCH_CPU_RESULT='+json.dumps(result,separators=(',',':')),flush=True); return result

def dino_mode(ink_gate):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    MODEL='facebook/dinov3-vitb16-pretrain-lvd1689m'; REV='5931719e67bbdb9737e363e781fb0c67687896bc'; token=os.getenv('HF_TOKEN')
    spec=json.loads((ROOT/'replication1_manifest.json').read_text()); bank,hb=build_bank(spec,ink_gate,True)
    images=[]; meta=[]
    for r in bank:
        for x in r['items']:
            p=x[2]; im=Image.fromarray(255-p).convert('RGB').resize((224,224),Image.Resampling.BILINEAR); images.append(im); meta.append((r['family'],r['id']))
    proc=AutoImageProcessor.from_pretrained(MODEL,revision=REV,token=token); model=AutoModel.from_pretrained(MODEL,revision=REV,token=token,torch_dtype=torch.float16,low_cpu_mem_usage=True).eval().cuda(); em=[]; B=64
    with torch.inference_mode():
        for i in range(0,len(images),B):
            x=proc(images=images[i:i+B],return_tensors='pt').to('cuda')
            with torch.autocast(device_type='cuda',dtype=torch.float16): y=model(**x).last_hidden_state[:,0].float()
            y=torch.nn.functional.normalize(y,dim=1); em.append(y.cpu().numpy()); print(json.dumps({'event':'dino_batch','done':min(i+B,len(images)),'total':len(images)}),flush=True)
    E=np.concatenate(em); by=defaultdict(list)
    for i,k in enumerate(meta): by[k].append(i)
    rows=[]
    for (fam,id_),ix in by.items():
        z=E[ix].mean(0); z=z/max(np.linalg.norm(z),1e-12); rows.append((fam,id_,z,len(ix)))
    ctrl=[r for r in rows if r[0]!='VOYNICH']; v=[r for r in rows if r[0]=='VOYNICH'][0]; labs=np.array([r[0] for r in ctrl],object); M=np.stack([r[2] for r in ctrl]); q=v[2]
    def cdist(a,b): return float(1-np.dot(a,b))
    cg=M[labs=='GREEK'].mean(0); cg/=np.linalg.norm(cg); cl=M[labs=='LATIN'].mean(0); cl/=np.linalg.norm(cl); A=cdist(q,cl)-cdist(q,cg); vals=[]; n=len(labs)
    for gix in itertools.combinations(range(n),n//2):
        g=np.zeros(n,bool); g[list(gix)]=True
        if np.array_equal(g,labs=='GREEK'): continue
        a=M[g].mean(0);a/=np.linalg.norm(a);b=M[~g].mean(0);b/=np.linalg.norm(b); vals.append(cdist(q,b)-cdist(q,a))
    vals=np.array(vals); mu=float(vals.mean());sd=float(vals.std(ddof=1));z=float((A-mu)/sd);p=float((1+np.sum(vals>=A))/(len(vals)+1)); result={'protocol':'2026-09-01.greek-affinity.final-fixedpatch.v1','mode':'dino','model':MODEL,'revision':REV,'n_embeddings':len(E),'distances':{'GREEK':cdist(q,cg),'LATIN':cdist(q,cl)},'greek_advantage':A,'null_mean':mu,'null_sd':sd,'z':z,'exact_p_one_sided':p,'null_assignments':len(vals),'decision':'DINO_PASS' if z>=2 and A>0 else 'DINO_NOT_PASS'}; print('FINAL_FIXEDPATCH_DINO_RESULT='+json.dumps(result,separators=(',',':')),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['cpu','dino'],default='cpu'); ap.add_argument('--strict',action='store_true'); ap.add_argument('--no-hands',action='store_true'); args=ap.parse_args(); gate=STRICT_INK if args.strict else BASE_INK
    cpu_mode(gate,not args.no_hands) if args.mode=='cpu' else dino_mode(gate)
