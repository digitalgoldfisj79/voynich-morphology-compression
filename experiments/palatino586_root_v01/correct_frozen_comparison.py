#!/usr/bin/env python3
import io,json,base64,hashlib,urllib.parse,re
import numpy as np, requests
from PIL import Image,ImageOps,ImageDraw
SB='https://ymaqlcfjmdwncdbjprmw.supabase.co'
NPZ=SB+'/storage/v1/object/public/bridge/p586_root_v01/embeddings/all_roots_dinov3_vit7b16_recovered.npz'
CP=SB+'/storage/v1/object/public/bridge/p586_root_v01/full_run/checkpoint.json'
UP=SB+'/functions/v1/p586-root-resume-upload-v01'
ST=SB+'/storage/v1/object/public/manuscripts/'
SEED=20260803
S=requests.Session();S.headers['User-Agent']='VoynichRootResearch/0.1-final-correction'
def get(u):
 r=S.get(u,timeout=180);r.raise_for_status();return r
def upload(path,ctype,data):
 r=S.post(UP,json={'path':path,'content_type':ctype,'data_b64':base64.b64encode(data).decode()},timeout=300);r.raise_for_status();print('UPLOAD',r.text,flush=True);return r.json()
def sha(b):return hashlib.sha256(b).hexdigest()
def img(path):return ImageOps.exif_transpose(Image.open(io.BytesIO(get(ST+urllib.parse.quote(path,safe='/')).content))).convert('RGB')
def jpg(im):
 b=io.BytesIO();im.save(b,'JPEG',quality=88,optimize=True);return b.getvalue()
cp_raw=get(CP).content;cp=json.loads(cp_raw);props=list(cp['proposals'])
late=[('c080_p00_r00','accept'),('c080_p02_r01','partial'),('c081_p00_r00','partial'),('c081_p01_r01','accept'),('c081_p02_r02','partial'),('c081_p03_r03','reject'),('c082_p01_r00','accept'),('c084_p00_r00','accept'),('c084_p01_r01','reject')]
for stem,status in late:
 c,p,r=[int(x) for x in re.fullmatch(r'c(\d+)_p(\d+)_r(\d+)',stem).groups()]
 props.append({'canvas_index':c,'plant_index':p,'root_index':r,'qa_status':status,'crop_path':f'p586_root_v01/full_run/crops/{stem}.png','recovery':'count-constrained from original job logs and focused QA'})
counts={s:sum(p['qa_status']==s for p in props) for s in ['accept','partial','reject']}
assert counts=={'accept':30,'partial':107,'reject':41},counts
frozen_paths={p['crop_path']:p['qa_status'] for p in props if p['qa_status'] in ('accept','partial')}
assert len(frozen_paths)==137,len(frozen_paths)
paths_blob=json.dumps({'protocol':'P586-VMS-ROOT-0.1-20260803','source_checkpoint_sha256':sha(cp_raw),'counts':counts,'retained_count':137,'paths':frozen_paths},sort_keys=True,indent=2).encode();upload('p586_root_v01/final/frozen_retained_paths.json','application/json',paths_blob)
raw=get(NPZ).content;z=np.load(io.BytesIO(raw),allow_pickle=True);X=z['embeddings'].astype('float32');keys=z['keys'];corpora=z['corpora'];groups=z['groups'];statuses=z['statuses'];paths=z['paths']
keep=np.array([(c!='p586') or (str(p) in frozen_paths) for c,p in zip(corpora,paths)])
X=X[keep];keys=keys[keep];corpora=corpora[keep];groups=groups[keep];statuses=statuses[keep];paths=paths[keep]
for i,(c,p) in enumerate(zip(corpora,paths)):
 if c=='p586': statuses[i]=frozen_paths[str(p)]
X/=np.linalg.norm(X,axis=1,keepdims=True)+1e-12
assert sum(corpora=='p586')==137
assert sum((corpora=='p586')&(statuses=='accept'))==30
assert sum((corpora=='p586')&(statuses=='partial'))==107
vi=np.where(corpora=='voynich')[0];bi=np.where(corpora=='bsb1784')[0]
def reciprocal(a,b):
 s=X[a]@X[b].T;ab=np.argmax(s,1);ba=np.argmax(s,0);return [(int(a[i]),int(b[j])) for i,j in enumerate(ab) if ba[j]==i]
def analyse(include_partial,seed):
 rng=np.random.default_rng(seed);pi=np.where((corpora=='p586')&((statuses=='accept')|(include_partial&(statuses=='partial'))))[0]
 target=np.r_[pi,bi];labels=np.r_[np.ones(len(pi),int),np.zeros(len(bi),int)];sim=X[vi]@X[target].T
 order=np.argsort(-sim,axis=1);tops={k:order[:,:min(k,len(target))] for k in (1,5,10)}
 sp=X[vi]@X[pi].T;sb=X[vi]@X[bi].T;dp=sp.max(1);db=sb.max(1)
 m={f'palatino_share_top{k}':float(labels[tops[k]].mean()) for k in (1,5,10)}
 m.update(mean_best_palatino=float(dp.mean()),mean_best_bsb1784=float(db.mean()),mean_best_difference=float((dp-db).mean()),median_best_difference=float(np.median(dp-db)),reciprocal_voynich_p586=len(reciprocal(vi,pi)),reciprocal_voynich_bsb1784=len(reciprocal(vi,bi)))
 obs={'top1':m['palatino_share_top1'],'top5':m['palatino_share_top5'],'top10':m['palatino_share_top10'],'diff':m['mean_best_difference']};null={k:np.empty(10000,np.float32) for k in obs}
 for q in range(10000):
  lab=np.zeros(len(target),int);lab[rng.choice(len(target),len(pi),replace=False)]=1
  for k in (1,5,10):null[f'top{k}'][q]=lab[tops[k]].mean()
  a=np.where(lab==1)[0];b=np.where(lab==0)[0];null['diff'][q]=(sim[:,a].max(1)-sim[:,b].max(1)).mean()
 ns={k:{'mean':float(v.mean()),'sd':float(v.std(ddof=1)),'p_upper':float((1+(v>=obs[k]-1e-12).sum())/10001),'p_lower':float((1+(v<=obs[k]+1e-12).sum())/10001)} for k,v in null.items()}
 vg=np.unique(groups[vi]);pg=np.unique(groups[pi]);bg=np.unique(groups[bi]);boot=np.empty(3000,np.float32)
 for q in range(3000):
  vv=np.concatenate([vi[groups[vi]==g] for g in rng.choice(vg,len(vg),replace=True)]);pp=np.concatenate([pi[groups[pi]==g] for g in rng.choice(pg,len(pg),replace=True)]);bb=np.concatenate([bi[groups[bi]==g] for g in rng.choice(bg,len(bg),replace=True)])
  boot[q]=((X[vv]@X[pp].T).max(1)-(X[vv]@X[bb].T).max(1)).mean()
 m['group_bootstrap_difference_ci95']=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]
 size=min(len(pi),len(bi));md=np.empty(10000,np.float32);wf=np.empty(10000,np.float32)
 for q in range(10000):
  pp=rng.choice(pi,size,replace=False);bb=rng.choice(bi,size,replace=False);d=(X[vi]@X[pp].T).max(1)-(X[vi]@X[bb].T).max(1);md[q]=d.mean();wf[q]=(d>0).mean()
 m['matched_pool_audit']={'secondary_posthoc':True,'reference_size_each':size,'repetitions':10000,'mean_best_difference':float(md.mean()),'reference_subsample_interval':[float(np.quantile(md,.025)),float(np.quantile(md,.975))],'voynich_query_win_fraction':float(wf.mean()),'win_fraction_interval':[float(np.quantile(wf,.025)),float(np.quantile(wf,.975))]}
 flat=np.argsort(-sp.ravel());pairs=[];used=set()
 for f in flat:
  a,b=np.unravel_index(int(f),sp.shape)
  if a in used:continue
  used.add(a);score=float(sp[a,b]);d=int(np.argmin(np.abs(sb[a]-score)))
  pairs.append({'voynich_index':int(vi[a]),'p586_index':int(pi[b]),'bsb1784_index':int(bi[d]),'voynich_key':str(keys[vi[a]]),'p586_key':str(keys[pi[b]]),'bsb1784_key':str(keys[bi[d]]),'voynich_path':str(paths[vi[a]]),'p586_path':str(paths[pi[b]]),'bsb1784_path':str(paths[bi[d]]),'p586_similarity':score,'bsb1784_similarity':float(sb[a,d])})
  if len(pairs)>=12:break
 return {'include_partial':include_partial,'counts':{'voynich':len(vi),'p586':len(pi),'bsb1784':len(bi)},'metrics':m,'null':ns,'top_pairs':pairs}
primary=analyse(False,SEED);sensitivity=analyse(True,SEED+1)
result={'protocol_id':'P586-VMS-ROOT-0.1-20260803','run_id':'ef6a0302-8269-4166-8c1c-63b14abb9c47','correction':'Uses original frozen checkpoint plus nine terminal decisions recovered from failed-job logs. Excludes nondeterministic replay of canvas 67 that inflated recovered sensitivity set from 137 to 139.','source_checkpoint_sha256':sha(cp_raw),'source_embedding_bundle_sha256':sha(raw),'embedding_model':str(z['model'][0]),'embedding_revision':str(z['revision'][0]),'embedding_dim':int(X.shape[1]),'proposal_counts':counts,'primary_accept_only':primary,'sensitivity_accept_plus_partial':sensitivity,'limitations':['First-pass model-localised and model-QAed crops; no full human crop adjudication.','BSB 1784 is one control manuscript, not a complete null panel.','Similarity does not establish exemplar, descent, provenance or botanical identity.']}
canon=json.dumps(result,sort_keys=True,separators=(',',':')).encode();result['result_sha256']=sha(canon);upload('p586_root_v01/final/corrected_comparison.json','application/json',json.dumps(result,indent=2).encode())
rng=np.random.default_rng(SEED+2);pairs=sensitivity['top_pairs'];sheet=Image.new('RGB',(1200,360*len(pairs)),'white');d=ImageDraw.Draw(sheet);key=[]
for t,p in enumerate(pairs):
 ims=[img(p['voynich_path']),img(p['p586_path']),img(p['bsb1784_path'])];flip=bool(rng.integers(0,2));A,B=(ims[2],ims[1]) if flip else (ims[1],ims[2]);key.append({'trial':t+1,'A':'bsb1784' if flip else 'p586','B':'p586' if flip else 'bsb1784',**p})
 for col,(lab,im) in enumerate([('QUERY',ims[0]),('A',A),('B',B)]):
  x=col*400;y=t*360;z=im.copy();z.thumbnail((370,300));sheet.paste(z,(x+(400-z.width)//2,y+45+(300-z.height)//2));d.rectangle((x,y,x+399,y+359),outline='black');d.text((x+8,y+8),f'Trial {t+1} {lab}',fill='black')
u1=upload('p586_root_v01/final/corrected_blind_triptychs.jpg','image/jpeg',jpg(sheet));upload('p586_root_v01/final/corrected_blind_key.json','application/json',json.dumps(key,indent=2).encode())
print('RESULT_JSON='+json.dumps({'result_sha256':result['result_sha256'],'proposal_counts':counts,'primary':primary,'sensitivity':sensitivity,'audit_url':u1.get('public_url')},sort_keys=True),flush=True)
