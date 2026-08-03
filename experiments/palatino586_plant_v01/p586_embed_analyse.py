#!/usr/bin/env python3
"""Frozen DINOv3 embedding and manuscript-level analysis. Run only after channel manifests freeze."""
from __future__ import annotations
import base64,gc,hashlib,io,json,os,time
from collections import defaultdict
import numpy as np,requests,torch
from PIL import Image
from transformers import AutoModel

PROTOCOL="P586-VMS-PLANT-0.1-20260803";SEED=20260803
MODEL="facebook/dinov3-vit7b16-pretrain-lvd1689m";REV="b80367753773648a6793235ab9c65cdbb029506f"
SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co";BRIDGE=SUPA+"/storage/v1/object/public/bridge/";UPLOAD=SUPA+"/functions/v1/p586-plant-upload-v01"
TARGET="bncf_palatino_586";VOYNICH="voynich"
MAIN=["bnf_lat_6862","herb_8510642bf2ce","herb_c51d653739d2","herb_0b5263630edb","herb_18f0aa144a2b","herb_eaec4fe75d89","herb_78e2bbc79062","herb_d91d01bd5276","bsb1784"]
KNOWN=["herb_7ce7efc90e6d","herb_fafef9a26da5","herb_205bfb89efbc"]
ALL=[TARGET,VOYNICH]+MAIN+KNOWN
S=requests.Session();S.headers["User-Agent"]="P586PlantDINO/0.1"

def get(u,tries=5):
 last=None
 for k in range(tries):
  try:r=S.get(u,timeout=240);r.raise_for_status();return r
  except Exception as e:last=e;time.sleep(min(15,1.7**k))
 raise RuntimeError(f"GET {u}: {last}")
def token():
 src=get("https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py").text
 import re
 return re.search(r'RUN_ID\s*=\s*"([^"]+)"',src).group(1)
TOKEN=token()
def upload(path,typ,data):
 p={"path":path,"content_type":typ,"data_b64":base64.b64encode(data).decode()};last=None
 for k in range(5):
  try:r=S.post(UPLOAD,headers={"x-upload-token":TOKEN},json=p,timeout=360);r.raise_for_status();return
  except Exception as e:last=e;time.sleep(min(15,1.7**k))
 raise RuntimeError(f"UPLOAD {path}: {last}")
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def prefix(cid):return "p586_plant_v01/target" if cid==TARGET else f"p586_plant_v01/controls/{cid}"
def load_manifests():
 out={};missing={}
 for cid in ALL:
  u=BRIDGE+prefix(cid)+"/channels_manifest_frozen.json"
  try:out[cid]=get(u).json()
  except Exception as e:missing[cid]=str(e)
 if TARGET not in out or VOYNICH not in out:raise RuntimeError(f"target/reference channel manifest missing: {missing}")
 return out,missing

def add_items(manifests):
 items=[]
 for cid,m in manifests.items():
  for p in m.get("plants",[]):
   broad=p.get("qa_status") in {"accept","partial"};strict=p.get("qa_status")=="accept";valid=p.get("mask_valid",False)
   base={"corpus":cid,"plant_id":p.get("plant_id"),"qa_status":p.get("qa_status"),"strict":strict,"broad":broad}
   if broad and p.get("crop_path"):items.append({**base,"kind":"whole_raw","path":p["crop_path"]})
   if broad and valid and p.get("masked_crop_path"):items.append({**base,"kind":"whole_masked","path":p["masked_crop_path"]})
   if broad and valid and p.get("above_strict_path"):items.append({**base,"kind":"above_strict","path":p["above_strict_path"]})
   if broad and valid and p.get("above_context_path"):items.append({**base,"kind":"above_context","path":p["above_context_path"]})
   for z in p.get("reproductive_structures",[]):
    if z.get("qa_status") not in {"accept","partial"} or not z.get("crop_path"):continue
    cls=z.get("qa_class") or z.get("proposed_class");items.append({**base,"kind":"reproductive","path":z["crop_path"],"repro_id":z.get("repro_id"),"repro_status":z.get("qa_status"),"repro_class":cls})
 for i,x in enumerate(items):x["item_id"]=f"i{i:06d}"
 return items

def preprocess(im):
 im=im.convert("RGB").resize((224,224),Image.Resampling.BILINEAR);a=np.asarray(im,dtype=np.float32)/255.0;a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32);return torch.from_numpy(a.transpose(2,0,1))
def embed(items):
 dev="cuda" if torch.cuda.is_available() else "cpu";model=AutoModel.from_pretrained(MODEL,revision=REV,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16 if dev=="cuda" else torch.float32).to(dev).eval();vec={};errors=[];batch=[];keys=[]
 def flush():
  if not batch:return
  x=torch.stack(batch).to(dev)
  with torch.inference_mode():o=model(pixel_values=x);v=o.last_hidden_state[:,0,:].float();v=v/torch.linalg.vector_norm(v,dim=1,keepdim=True).clamp_min(1e-12)
  for k,z in zip(keys,v.cpu().numpy()):vec[k]=z.astype(np.float32)
  batch.clear();keys.clear()
 for n,it in enumerate(items,1):
  try:raw=get(BRIDGE+it["path"]).content;im=Image.open(io.BytesIO(raw));batch.append(preprocess(im));keys.append(it["item_id"]);it["image_sha256"]=hashlib.sha256(raw).hexdigest()
  except Exception as e:errors.append({"item_id":it["item_id"],"path":it["path"],"error":f"{type(e).__name__}: {e}"})
  if len(batch)>=8:flush()
  if n%50==0:print(json.dumps({"event":"embed_progress","n":n,"total":len(items),"vectors":len(vec),"errors":len(errors)}),flush=True)
 flush();del model;gc.collect();torch.cuda.empty_cache();return vec,errors

def variants(items,vec):
 def ok(it,name):
  if it["item_id"] not in vec:return False
  if name=="whole_unmasked_strict":return it["kind"]=="whole_raw" and it["strict"]
  if name=="whole_unmasked_broad":return it["kind"]=="whole_raw" and it["broad"]
  if name=="whole_masked_strict":return it["kind"]=="whole_masked" and it["strict"]
  if name=="whole_masked_broad":return it["kind"]=="whole_masked" and it["broad"]
  if name=="above_strict_crop_strict":return it["kind"]=="above_strict" and it["strict"]
  if name=="above_strict_crop_broad":return it["kind"]=="above_strict" and it["broad"]
  if name=="above_context_strict":return it["kind"]=="above_context" and it["strict"]
  if name=="above_context_broad":return it["kind"]=="above_context" and it["broad"]
  if name=="flowers_strict":return it["kind"]=="reproductive" and it.get("repro_status")=="accept" and it.get("repro_class") in {"flower","flower_head","inflorescence"}
  if name=="flowers_broad":return it["kind"]=="reproductive" and it.get("repro_status") in {"accept","partial"} and it.get("repro_class") in {"flower","flower_head","inflorescence","bud"}
  if name=="fruit_seed_broad":return it["kind"]=="reproductive" and it.get("repro_status") in {"accept","partial"} and it.get("repro_class") in {"fruit","seed_head"}
  return False
 names=["whole_unmasked_strict","whole_unmasked_broad","whole_masked_strict","whole_masked_broad","above_strict_crop_strict","above_strict_crop_broad","above_context_strict","above_context_broad","flowers_strict","flowers_broad","fruit_seed_broad"]
 out={}
 for name in names:
  d=defaultdict(list);meta=defaultdict(list)
  for it in items:
   if ok(it,name):d[it["corpus"]].append(vec[it["item_id"]]);meta[it["corpus"]].append(it)
  out[name]={cid:np.stack(v) for cid,v in d.items() if v};out[name+"__meta"]=dict(meta)
 return out

def best(q,r):return (q@r.T).max(axis=1)
def analyse_variant(name,data,meta,rng):
 counts={c:len(v) for c,v in data.items()};eligible=[c for c in MAIN if c in data and len(data[c])>=8];out={"counts":counts,"eligible_controls":eligible}
 if TARGET not in data or VOYNICH not in data or len(data[TARGET])<1 or len(data[VOYNICH])<1:return out
 full={}
 for c,v in data.items():
  if c==VOYNICH:continue
  b=best(v,data[VOYNICH]);full[c]={"mean_best":float(b.mean()),"median_best":float(np.median(b)),"n":len(v)}
 out["full_scores"]=full
 participants=[TARGET]+eligible
 if not eligible:return out
 m=min([len(data[VOYNICH])]+[len(data[c]) for c in participants]);out["balanced_n"]=m;reps={c:[] for c in participants}
 for _ in range(500):
  rr=data[VOYNICH][rng.choice(len(data[VOYNICH]),m,replace=False)]
  for c in participants:
   qq=data[c][rng.choice(len(data[c]),m,replace=False)];reps[c].append(float(best(qq,rr).mean()))
 means={c:float(np.mean(v)) for c,v in reps.items()};out["balanced_mean_best"]=means;effect=means[TARGET]-float(np.mean([means[c] for c in eligible]));out["primary_effect_target_minus_control_mean"]=effect
 boot=[]
 for _ in range(10000):
  tv=reps[TARGET][rng.integers(500)];cs=rng.choice(eligible,len(eligible),replace=True);cv=np.mean([reps[c][rng.integers(500)] for c in cs]);boot.append(tv-cv)
 out["hierarchical_bootstrap_95ci"]=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))];out["manuscript_label_rank_p_upper"]=(1+sum(means[c]>=means[TARGET] for c in eligible))/(1+len(eligible));out["target_rank_descending"]=1+sum(means[c]>means[TARGET] for c in eligible)
 out["leave_one_control_out"]={c:means[TARGET]-float(np.mean([means[d] for d in eligible if d!=c])) if len(eligible)>1 else None for c in eligible}
 # nearest-source shares from a balanced candidate pool for each Voynich item
 shares=defaultdict(list);topks=(1,5,10)
 for _ in range(100):
  pools=[];labs=[]
  for c in participants:
   x=data[c][rng.choice(len(data[c]),m,replace=False)];pools.append(x);labs.extend([c]*m)
  pool=np.concatenate(pools);labs=np.array(labs);sim=data[VOYNICH]@pool.T;order=np.argsort(-sim,axis=1)
  for k in topks:
   flat=labs[order[:,:min(k,order.shape[1])]].reshape(-1)
   for c in participants:shares[(k,c)].append(float(np.mean(flat==c)))
 out["topk_source_shares"]={str(k):{c:float(np.mean(shares[(k,c)])) for c in participants} for k in topks}
 # strongest target pairs for blind sheets
 sim=data[TARGET]@data[VOYNICH].T;pairs=[]
 for qi,ri in sorted(((i,int(np.argmax(sim[i]))) for i in range(len(data[TARGET]))),key=lambda z:sim[z[0],z[1]],reverse=True)[:12]:pairs.append({"target_index":qi,"voynich_index":ri,"similarity":float(sim[qi,ri]),"target_item":meta[TARGET][qi],"voynich_item":meta[VOYNICH][ri]})
 out["strongest_target_pairs"]=pairs
 return out

def known_answer(vs):
 pairs=[("herb_7ce7efc90e6d","herb_fafef9a26da5","exact_registry_duplicate"),("herb_18f0aa144a2b","herb_205bfb89efbc","same_manuscript_alt_manifest")];out={}
 for name in ["whole_masked_broad","above_context_broad"]:
  d=vs.get(name,{});r=[]
  for a,b,label in pairs:
   if a in d and b in d:r.append({"pair":label,"a":a,"b":b,"a_to_b_mean_best":float(best(d[a],d[b]).mean()),"b_to_a_mean_best":float(best(d[b],d[a]).mean()),"n_a":len(d[a]),"n_b":len(d[b])})
  out[name]=r
 return out

def persist(items,vec):
 by=defaultdict(list)
 for it in items:
  if it["item_id"] in vec:by[it["corpus"]].append(it)
 index=[]
 for cid,rows in by.items():
  ids=np.array([r["item_id"] for r in rows]);mat=np.stack([vec[x] for x in ids]);b=io.BytesIO();np.savez_compressed(b,item_id=ids,embedding=mat.astype(np.float32));path=f"p586_plant_v01/embeddings/{cid}_dinov3_vit7b16_cls_f32.npz";upload(path,"application/octet-stream",b.getvalue());index.append({"corpus":cid,"path":path,"n":len(rows),"sha256":hashlib.sha256(b.getvalue()).hexdigest()})
 return index

def main():
 mans,missing=load_manifests();items=add_items(mans);vec,errors=embed(items);index=persist(items,vec);vs=variants(items,vec);rng=np.random.default_rng(SEED);results={}
 for name in [k for k in vs if not k.endswith("__meta")]:results[name]=analyse_variant(name,vs[name],vs.get(name+"__meta",{}),rng)
 report={"protocol_id":PROTOCOL,"model_id":MODEL,"model_revision":REV,"representation":"CLS float32 L2-normalized","preprocessing":{"resize":[224,224],"resample":"bilinear","mean":[.485,.456,.406],"std":[.229,.224,.225],"rescale":1/255},"seed":SEED,"channel_manifests":{c:m.get("channel_manifest_sha256") for c,m in mans.items()},"missing_manifests":missing,"embedding_items":len(vec),"embedding_errors":errors,"embedding_files":index,"results":results,"known_answer_controls":known_answer(vs)};report["result_sha256"]=csha(report);upload("p586_plant_v01/results/dinov3_analysis.json","application/json",json.dumps(report,indent=2,sort_keys=True).encode());upload("p586_plant_v01/results/blind_pair_candidates.json","application/json",json.dumps({k:v.get("strongest_target_pairs",[]) for k,v in results.items()},indent=2,sort_keys=True).encode());print("RESULT_JSON="+json.dumps({"result_sha256":report["result_sha256"],"embedding_items":len(vec),"missing":missing,"summary":{k:{x:v.get(x) for x in ("counts","eligible_controls","primary_effect_target_minus_control_mean","hierarchical_bootstrap_95ci","manuscript_label_rank_p_upper","target_rank_descending")} for k,v in results.items()}},sort_keys=True),flush=True)
if __name__=="__main__":main()
