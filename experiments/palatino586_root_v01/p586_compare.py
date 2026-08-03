#!/usr/bin/env python3
import os, io, json, time, base64, hashlib, math, random, warnings, csv
import numpy as np
import requests
import torch
from PIL import Image, ImageOps, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings('ignore')
SEED=20260803
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
RUN_ID=os.environ.get('RUN_ID','unknown')
LOCATOR_RUN=os.environ['LOCATOR_RUN_ID']
MODEL_ID='facebook/dinov3-vitl16-pretrain-lvd1689m'
SUPABASE='https://ymaqlcfjmdwncdbjprmw.supabase.co'
ANON='sb_publishable_BOm91KbAPOZDCQ7H3yLFzw_VtNPk2ap'
STORAGE=SUPABASE+'/storage/v1/object/public/manuscripts/'
BRIDGE=SUPABASE+'/storage/v1/object/public/bridge/'
UPLOAD_ENDPOINT=SUPABASE+'/functions/v1/p586-root-upload-v01'
PREFIX='p586_root_v01/comparison_'+RUN_ID
ACCEPTED_URL=BRIDGE+f'p586_root_v01/qwen_locator_{LOCATOR_RUN}/accepted_manifest.json'
ALL_LOCATOR_URL=BRIDGE+f'p586_root_v01/qwen_locator_{LOCATOR_RUN}/locator_results.json'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
S=requests.Session(); S.headers.update({'User-Agent':'VoynichRootResearch/0.1'})


def log(event,**kw):print(json.dumps({'event':event,'time':time.time(),**kw},sort_keys=True),flush=True)
def get(url,tries=5,timeout=120,headers=None,params=None):
    last=None
    for k in range(tries):
        try:
            r=S.get(url,timeout=timeout,headers=headers,params=params); r.raise_for_status(); return r
        except Exception as e:last=e;time.sleep(min(15,1.5**k+random.random()))
    raise RuntimeError(f'GET failed {url}: {last}')
def load_image(url):return ImageOps.exif_transpose(Image.open(io.BytesIO(get(url).content))).convert('RGB')
def upload(relpath,ctype,data):
    payload={'path':f'{PREFIX}/{relpath}','content_type':ctype,'data_b64':base64.b64encode(data).decode()}
    r=S.post(UPLOAD_ENDPOINT,headers={'x-upload-token':os.environ['UPLOAD_TOKEN'],'content-type':'application/json'},json=payload,timeout=180);r.raise_for_status();out=r.json();log('upload',path=out.get('path'),bytes=out.get('bytes'));return out

def fetch_objects(ms,parts):
    url=SUPABASE+'/rest/v1/herbal_objects'
    params={'select':'id,manuscript_id,slug,obj_index,part,crop_path,crop_qa','manuscript_id':f'eq.{ms}','part':'in.('+','.join(parts)+')','crop_path':'not.is.null','limit':'1000'}
    rows=get(url,headers={'apikey':ANON},params=params).json()
    return [r for r in rows if r.get('crop_qa') not in ('spurious_fragment','genuine_miss')]
def feature_batch(model,proc,images,batch=24):
    feats=[]
    for s in range(0,len(images),batch):
        inp=proc(images=images[s:s+batch],return_tensors='pt').to(DEVICE)
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=DEVICE=='cuda'):
            h=model(**inp).last_hidden_state.float();f=torch.cat([h[:,0],h[:,1:].mean(1),h[:,1:].amax(1)],dim=1);f=torch.nn.functional.normalize(f,dim=1)
        feats.append(f.cpu().numpy().astype('float32'));log('embed',done=min(s+batch,len(images)),total=len(images))
    return np.concatenate(feats)
def ci_boot(values,B=10000):
    rng=np.random.default_rng(SEED);v=np.asarray(values,float);means=np.empty(B)
    for i in range(B):means[i]=rng.choice(v,len(v),replace=True).mean()
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
def matched_test(P,V,BX,reps=10000):
    rng=np.random.default_rng(SEED);m=min(len(V),len(BX));obs=[];null=[];win=[]
    for _ in range(reps):
        vi=rng.choice(len(V),m,replace=False); pool=np.concatenate([V[vi],BX],axis=0)
        sv=P@pool.T; a=sv[:,:m]; b=sv[:,m:]
        o=float((np.sort(a,axis=1)[:,-3:].mean(1)-np.sort(b,axis=1)[:,-3:].mean(1)).mean());obs.append(o)
        win.append(float((a.max(1)>b.max(1)).mean()))
        perm=rng.permutation(2*m); aa=sv[:,perm[:m]];bb=sv[:,perm[m:]]
        null.append(float((np.sort(aa,axis=1)[:,-3:].mean(1)-np.sort(bb,axis=1)[:,-3:].mean(1)).mean()))
    obs=np.array(obs);null=np.array(null)
    return {'matched_reference_size':m,'repetitions':reps,'mean_top3_delta':float(obs.mean()),'delta_interval':[float(np.quantile(obs,.025)),float(np.quantile(obs,.975))],
            'nearest_voynich_fraction':float(np.mean(win)),'nearest_voynich_interval':[float(np.quantile(win,.025)),float(np.quantile(win,.975))],
            'permutation_p':float((1+np.sum(null>=obs))/(1+reps)),'null_mean':float(null.mean()),'null_sd':float(null.std(ddof=1))}
def pair_sheet(rows,start,count=6):
    batch=rows[start:start+count];cols=3;cw=330;ch=350;sheet=Image.new('RGB',(cols*cw,len(batch)*ch),'white');d=ImageDraw.Draw(sheet)
    for ri,r in enumerate(batch):
        y=ri*ch;ims=[r['pal_image'],r['v_image'],r['b_image']];labels=['Palatino',f"Voynich {r['v_label']} {r['v_sim']:.3f}",f"BSB {r['b_label']} {r['b_sim']:.3f}"]
        for j,(im,lab) in enumerate(zip(ims,labels)):
            x=j*cw;z=im.copy();z.thumbnail((310,280));sheet.paste(z,(x+(cw-z.width)//2,y+45));d.rectangle((x,y,x+cw-1,y+ch-1),outline='black',width=1);d.text((x+8,y+8),lab,fill='black')
        d.text((8,y+25),f"canvas {r['page']:03d} {r['qa_label']} delta {r['v_sim']-r['b_sim']:.3f}",fill='black')
    bio=io.BytesIO();sheet.save(bio,'JPEG',quality=90,optimize=True);return bio.getvalue()

protocol={'protocol_id':'P586-VOYNICH-ROOT-COMPARE-V0.1-20260803','run_id':RUN_ID,'seed':SEED,'model_id':MODEL_ID,'locator_run':LOCATOR_RUN,
          'primary_set':'CLEAR_ROOT only if n>=10, otherwise CLEAR_ROOT+PARTIAL_ROOT','control':'BSB Cgm 728 roots, matched to 25 Voynich roots per repetition','repetitions':10000,
          'claim_limit':'exploratory morphological enrichment only; no exemplar/source inference'}
upload('protocol.json','application/json',json.dumps(protocol,indent=2,sort_keys=True).encode())
accepted=get(ACCEPTED_URL).json();all_locator=get(ALL_LOCATOR_URL).json()
clear=[r for r in accepted if r.get('qa_label')=='CLEAR_ROOT'];primary=clear if len(clear)>=10 else accepted
rejected=[r for r in all_locator if not r.get('accepted')]
log('palatino_sets',accepted=len(accepted),clear=len(clear),primary=len(primary),rejected=len(rejected))

vrows=fetch_objects('voynich',['root','plant']);brows=fetch_objects('bsb1784',['root'])
vroot=[r for r in vrows if r['part']=='root'];vplant=[r for r in vrows if r['part']=='plant']
log('reference_sets',vroot=len(vroot),vplant=len(vplant),bsb=len(brows))

pimgs=[];pmeta=[]
for r in primary:
    try:pimgs.append(load_image(r['crop_url']));pmeta.append(r)
    except Exception as e:log('pal_fail',page=r.get('page'),error=str(e))
rimgs=[];rmeta=[]
for r in rejected:
    try:rimgs.append(load_image(r['crop_url']));rmeta.append(r)
    except Exception:pass
vri=[];vrm=[]
for r in vroot:
    try:vri.append(load_image(STORAGE+r['crop_path']));vrm.append(r)
    except Exception as e:log('vroot_fail',path=r['crop_path'],error=str(e))
vpi=[];vpm=[]
for r in vplant:
    try:vpi.append(load_image(STORAGE+r['crop_path']));vpm.append(r)
    except Exception:pass
bi=[];bm=[]
for r in brows:
    try:bi.append(load_image(STORAGE+r['crop_path']));bm.append(r)
    except Exception as e:log('bsb_fail',path=r['crop_path'],error=str(e))

proc=AutoImageProcessor.from_pretrained(MODEL_ID);model=AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
PX=feature_batch(model,proc,pimgs);VX=feature_batch(model,proc,vri);BX=feature_batch(model,proc,bi);VPX=feature_batch(model,proc,vpi)
RX=feature_batch(model,proc,rimgs) if rimgs else np.empty((0,PX.shape[1]),dtype='float32')

X=np.concatenate([VX,VPX]);y=np.array([1]*len(VX)+[0]*len(VPX));groups=np.array([r['slug'] for r in vrm+vpm])
clf=LogisticRegression(C=.1,max_iter=2000,class_weight='balanced',solver='liblinear',random_state=SEED)
auc=cross_val_score(clf,X,y,groups=groups,cv=GroupKFold(5),scoring='roc_auc');clf.fit(X,y)
p_root=clf.predict_proba(PX)[:,1];r_root=clf.predict_proba(RX)[:,1] if len(RX) else np.array([])

pv=PX@VX.T;pb=PX@BX.T
p_v_top1=pv.max(1);p_b_top1=pb.max(1);delta=p_v_top1-p_b_top1
matched=matched_test(PX,VX,BX,10000)

pair_rows=[]
for i,r in enumerate(pmeta):
    vi=int(np.argmax(pv[i]));bi_=int(np.argmax(pb[i]));pair_rows.append({'page':int(r['page']),'qa_label':r['qa_label'],'pal_url':r['crop_url'],'v_label':f"{vrm[vi]['slug']}#{vrm[vi]['obj_index']}",'v_url':STORAGE+vrm[vi]['crop_path'],'v_sim':float(pv[i,vi]),'b_label':f"{bm[bi_]['slug']}#{bm[bi_]['obj_index']}",'b_url':STORAGE+bm[bi_]['crop_path'],'b_sim':float(pb[i,bi_]),'root_probability':float(p_root[i]),'pal_image':pimgs[i],'v_image':vri[vi],'b_image':bi[bi_]})
pair_rows.sort(key=lambda r:r['v_sim']-r['b_sim'],reverse=True)
for s in range(0,len(pair_rows),6):upload(f'pair_sheets/pairs_{s:03d}-{min(s+5,len(pair_rows)-1):03d}.jpg','image/jpeg',pair_sheet(pair_rows,s,6))

serial_pairs=[{k:v for k,v in r.items() if not k.endswith('_image')} for r in pair_rows]
report={'protocol':protocol,'counts':{'palatino_primary':len(PX),'palatino_clear':len(clear),'palatino_all_accepted':len(accepted),'palatino_rejected_available':len(RX),'voynich_roots':len(VX),'voynich_plants':len(VPX),'bsb_roots':len(BX)},
        'root_validity':{'voynich_grouped_auc_mean':float(auc.mean()),'voynich_grouped_auc_sd':float(auc.std()),'palatino_median_probability':float(np.median(p_root)),'palatino_mean_probability':float(np.mean(p_root)),'rejected_median_probability':float(np.median(r_root)) if len(r_root) else None},
        'similarity':{'palatino_to_voynich_top1_mean':float(p_v_top1.mean()),'palatino_to_bsb_top1_mean':float(p_b_top1.mean()),'top1_delta_mean':float(delta.mean()),'top1_delta_ci':ci_boot(delta),'voynich_win_fraction':float((delta>0).mean()),'matched_test':matched},
        'pairs':serial_pairs}
blob=json.dumps(report,sort_keys=True,separators=(',',':')).encode();report['result_sha256']=hashlib.sha256(blob).hexdigest()
upload('comparison_report.json','application/json',json.dumps(report,indent=2,sort_keys=True).encode())
upload('nearest_pairs.json','application/json',json.dumps(serial_pairs,indent=2,sort_keys=True).encode())
sio=io.StringIO();w=csv.DictWriter(sio,fieldnames=['page','qa_label','v_label','v_sim','b_label','b_sim','root_probability']);w.writeheader();w.writerows([{k:r[k] for k in w.fieldnames} for r in serial_pairs]);upload('nearest_pairs.csv','text/csv',sio.getvalue().encode())
print('RESULT_JSON='+json.dumps({'run_id':RUN_ID,'prefix':PREFIX,'counts':report['counts'],'validity':report['root_validity'],'similarity':report['similarity'],'sha256':report['result_sha256']},sort_keys=True),flush=True)
