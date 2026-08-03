#!/usr/bin/env python3
"""Construct and adjudicate sealed A/B morphology sheets from frozen DINO results."""
from __future__ import annotations
import base64,hashlib,io,json,os,random,re,time
from collections import defaultdict
import numpy as np,requests,torch
from PIL import Image,ImageDraw,ImageOps

PROTOCOL="P586-VMS-PLANT-0.1-20260803";SEED=20260803;random.seed(SEED)
SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co";BRIDGE=SUPA+"/storage/v1/object/public/bridge/";EP=SUPA+"/functions/v1/p586-plant-upload-v01"
TARGET="bncf_palatino_586";VOYNICH="voynich";MAIN=["bnf_lat_6862","herb_8510642bf2ce","herb_c51d653739d2","herb_0b5263630edb","herb_18f0aa144a2b","herb_eaec4fe75d89","herb_78e2bbc79062","herb_d91d01bd5276","bsb1784"]
ALL=[TARGET,VOYNICH]+MAIN+["herb_7ce7efc90e6d","herb_fafef9a26da5","herb_205bfb89efbc"]
S=requests.Session();S.headers["User-Agent"]="P586BlindMorphology/0.1"
def get(u):r=S.get(u,timeout=240);r.raise_for_status();return r
def token():
 s=get("https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py").text;return re.search(r'RUN_ID\s*=\s*"([^"]+)"',s).group(1)
TOKEN=token()
def upload(path,typ,data):
 r=S.post(EP,headers={"x-upload-token":TOKEN},json={"path":path,"content_type":typ,"data_b64":base64.b64encode(data).decode()},timeout=360);r.raise_for_status()
def prefix(c):return "p586_plant_v01/target" if c==TARGET else f"p586_plant_v01/controls/{c}"
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def png(im):b=io.BytesIO();im.save(b,"PNG",optimize=True);return b.getvalue()

def manifests():
 out={}
 for c in ALL:
  try:out[c]=get(BRIDGE+prefix(c)+"/channels_manifest_frozen.json").json()
  except Exception:pass
 return out

def items(ms):
 out=[]
 for c in ALL:
  if c not in ms:continue
  for p in ms[c].get("plants",[]):
   broad=p.get("qa_status") in {"accept","partial"};strict=p.get("qa_status")=="accept";valid=p.get("mask_valid",False);base={"corpus":c,"plant_id":p.get("plant_id"),"strict":strict,"broad":broad}
   if broad and p.get("crop_path"):out.append({**base,"kind":"whole_raw","path":p["crop_path"]})
   if broad and valid and p.get("masked_crop_path"):out.append({**base,"kind":"whole_masked","path":p["masked_crop_path"]})
   if broad and valid and p.get("above_strict_path"):out.append({**base,"kind":"above_strict","path":p["above_strict_path"]})
   if broad and valid and p.get("above_context_path"):out.append({**base,"kind":"above_context","path":p["above_context_path"]})
   for z in p.get("reproductive_structures",[]):
    if z.get("qa_status") in {"accept","partial"} and z.get("crop_path"):out.append({**base,"kind":"reproductive","path":z["crop_path"],"repro_id":z.get("repro_id"),"repro_status":z.get("qa_status"),"repro_class":z.get("qa_class") or z.get("proposed_class")})
 for i,x in enumerate(out):x["item_id"]=f"i{i:06d}"
 return out

def load_vectors(it):
 v={};by=defaultdict(list)
 for x in it:by[x["corpus"]].append(x)
 for c in by:
  try:
   z=np.load(io.BytesIO(get(BRIDGE+f"p586_plant_v01/embeddings/{c}_dinov3_vit7b16_cls_f32.npz").content),allow_pickle=False)
   for k,a in zip(z["item_id"],z["embedding"]):v[str(k)]=a.astype(np.float32)
  except Exception:pass
 return v

def choose_variant(it,v,name):
 def ok(x):
  if x["item_id"] not in v:return False
  if name=="whole":return x["kind"]=="whole_masked" and x["broad"]
  if name=="above":return x["kind"]=="above_context" and x["broad"]
  if name=="flowers":return x["kind"]=="reproductive" and x.get("repro_status") in {"accept","partial"} and x.get("repro_class") in {"flower","flower_head","inflorescence","bud"}
  return False
 d=defaultdict(list)
 for x in it:
  if ok(x):d[x["corpus"]].append((x,v[x["item_id"]]))
 return d

def load_image(x):return Image.open(io.BytesIO(get(BRIDGE+x["path"]).content)).convert("RGB")
def sheet(anchor,a,b,title):
 W,H=1200,720;im=Image.new("RGB",(W,H),"white");d=ImageDraw.Draw(im);d.text((20,15),title,fill="black");slots=[(20,60,380,680),(420,60,780,680),(820,60,1180,680)];labs=["ANCHOR","A","B"]
 for src,sl,lab in zip((anchor,a,b),slots,labs):
  x0,y0,x1,y1=sl;q=ImageOps.contain(src,(x1-x0,y1-y0-35));im.paste(q,(x0+(x1-x0-q.width)//2,y0+30+(y1-y0-35-q.height)//2));d.text((x0+5,y0+5),lab,fill="black")
 return im

def infer_setup():
 base="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/1f57a841e79f5677726025a2ecb70fe5901e6cd5/experiments/palatino586_plant_v01/p586_target_extract.py";code=get(base).text;ns={"__name__":"lib"};exec(compile(code,base,"exec"),ns);H=ns["H"];P=H["AutoProcessor"].from_pretrained(ns["MODEL"],token=os.environ.get("HF_TOKEN"),min_pixels=256*28*28,max_pixels=1280*28*28);M=H["Qwen2_5_VLForConditionalGeneration"].from_pretrained(ns["MODEL"],token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval();return ns,H,M,P
PROMPT='''Image 1 is an anchor medieval plant or reproductive structure. Images 2 and 3 are candidates A and B. Without guessing manuscript identity, choose which candidate has the stronger specific visible morphological correspondence to the anchor in branching, leaf form, stem organisation, flower/seed structure and overall topology. Do not choose for beauty, colour, parchment or general drawing style. Return strict JSON only: {"decision":"A|B|tie|abstain","confidence":0.0,"reason":"brief"}. Use tie when equally plausible and abstain when malformed or not comparable.'''

def main():
 ms=manifests();it=items(ms);v=load_vectors(it);trials=[]
 for channel in ("whole","above","flowers"):
  d=choose_variant(it,v,channel)
  if TARGET not in d or VOYNICH not in d:continue
  tv=np.stack([z for _,z in d[TARGET]]);vv=np.stack([z for _,z in d[VOYNICH]]);sim=tv@vv.T;rank=[]
  for i in range(len(tv)):
   j=int(np.argmax(sim[i]));rank.append((float(sim[i,j]),i,j))
  used_t=set();used_v=set();used_c=set()
  for score,ti,vi in sorted(rank,reverse=True):
   if len([x for x in trials if x["channel"]==channel])>=6:break
   if ti in used_t or vi in used_v:continue
   anchor=d[VOYNICH][vi][0];target=d[TARGET][ti][0];av=v[anchor["item_id"]];cands=[]
   for c in MAIN:
    for x,z in d.get(c,[]):
     if x["item_id"] in used_c:continue
     s=float(z@av);cands.append((abs(s-score),s,c,x))
   if not cands:continue
   _,cs,cc,control=min(cands,key=lambda x:x[0]);used_t.add(ti);used_v.add(vi);used_c.add(control["item_id"]);target_side="A" if random.random()<.5 else "B"
   a=target if target_side=="A" else control;b=control if target_side=="A" else target
   trials.append({"trial_id":f"{channel}_{len([x for x in trials if x['channel']==channel])+1:02d}","channel":channel,"anchor":anchor,"candidate_A":a,"candidate_B":b,"target_side":target_side,"target_similarity":score,"control_similarity":cs,"control_corpus":cc})
 ns,H,M,P=infer_setup();results=[]
 for t in trials:
  ai=load_image(t["anchor"]);A=load_image(t["candidate_A"]);B=load_image(t["candidate_B"]);sh=sheet(ai,A,B,f"Blind morphology trial {t['trial_id']}");data=png(sh);path=f"p586_plant_v01/blind/{t['trial_id']}.png";upload(path,"image/png",data);ans,raw=ns["infer"](M,P,[ai,A,B],PROMPT,400);dec=str(ans.get("decision","abstain")) if isinstance(ans,dict) else "abstain";dec=dec if dec in {"A","B","tie","abstain"} else "abstain";selected="target" if dec==t["target_side"] else "control" if dec in {"A","B"} else dec;results.append({**t,"sheet_path":path,"sheet_sha256":hashlib.sha256(data).hexdigest(),"decision":dec,"selected":selected,"qa_raw":ans,"qa_response_raw":raw})
  print(json.dumps({"event":"blind_trial","trial":t["trial_id"],"decision":dec,"selected":selected},sort_keys=True),flush=True)
 summary={}
 for ch in ("whole","above","flowers"):
  q=[x for x in results if x["channel"]==ch];summary[ch]={"valid_trials":len(q),"target_selected":sum(x["selected"]=="target" for x in q),"control_selected":sum(x["selected"]=="control" for x in q),"ties":sum(x["selected"]=="tie" for x in q),"abstentions":sum(x["selected"]=="abstain" for x in q)}
 out={"protocol_id":PROTOCOL,"seed":SEED,"rule":"Voynich anchor; A/B is strongest Palatino match versus same-anchor similarity-matched manuscript control","trials":results,"summary":summary};out["result_sha256"]=csha(out);upload("p586_plant_v01/results/blind_adjudication.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode());upload("p586_plant_v01/results/blind_key.json","application/json",json.dumps([{"trial_id":x["trial_id"],"target_side":x["target_side"],"control_corpus":x["control_corpus"]} for x in results],indent=2,sort_keys=True).encode());print("RESULT_JSON="+json.dumps({"result_sha256":out["result_sha256"],"summary":summary},sort_keys=True),flush=True)
if __name__=="__main__":main()
