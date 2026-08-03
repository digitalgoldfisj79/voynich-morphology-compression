#!/usr/bin/env python3
"""Build the final human-readable evidence ZIP after analysis and blind review."""
from __future__ import annotations
import base64,csv,hashlib,io,json,os,re,zipfile
from collections import defaultdict
import requests
from PIL import Image,ImageDraw,ImageOps
SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co";BRIDGE=SUPA+"/storage/v1/object/public/bridge/";EP=SUPA+"/functions/v1/p586-plant-upload-v01";TICKET=SUPA+"/functions/v1/p586-plant-bundle-ticket-v01"
TARGET="bncf_palatino_586";IDS=[TARGET,"voynich","bnf_lat_6862","herb_8510642bf2ce","herb_c51d653739d2","herb_0b5263630edb","herb_18f0aa144a2b","herb_eaec4fe75d89","herb_78e2bbc79062","herb_d91d01bd5276","bsb1784","herb_7ce7efc90e6d","herb_fafef9a26da5","herb_205bfb89efbc"]
S=requests.Session();S.headers["User-Agent"]="P586PlantBundle/0.1"
def get(u):r=S.get(u,timeout=300);r.raise_for_status();return r
def token():
 s=get("https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py").text;return re.search(r'RUN_ID\s*=\s*"([^"]+)"',s).group(1)
TOKEN=token()
def prefix(c):return "p586_plant_v01/target" if c==TARGET else f"p586_plant_v01/controls/{c}"
def jget(path):return get(BRIDGE+path).json()
def add_json(z,name,obj):z.writestr(name,json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False))
def add_url(z,name,url):z.writestr(name,get(url).content)
def csv_bytes(rows,fields):
 b=io.StringIO();w=csv.DictWriter(b,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows);return b.getvalue().encode()
def contact(images,labels,title):
 cols=4;cw,ch=320,360;rows=max(1,(len(images)+cols-1)//cols);out=Image.new("RGB",(cols*cw,50+rows*ch),"white");d=ImageDraw.Draw(out);d.text((10,15),title,fill="black")
 for i,(im,lab) in enumerate(zip(images,labels)):
  x=(i%cols)*cw;y=50+(i//cols)*ch;q=ImageOps.contain(im,(cw-16,ch-55));out.paste(q,(x+(cw-q.width)//2,y));d.text((x+6,y+ch-45),lab[:48],fill="black")
 b=io.BytesIO();out.save(b,"JPEG",quality=88,optimize=True);return b.getvalue()
def main():
 analysis=jget("p586_plant_v01/results/dinov3_analysis.json");blind=jget("p586_plant_v01/results/blind_adjudication.json");buf=io.BytesIO();summary=[]
 with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  gh="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/gpt56/p586-plant-v01-20260803/experiments/palatino586_plant_v01/"
  for f in ["PREREGISTRATION.md","PREREGISTRATION_AMENDMENT_01.md","CONTROL_PANEL.json","DINOV3_FREEZE.json"]:add_url(z,"00_PROTOCOL/"+f,gh+f)
  add_json(z,"01_RESULTS/dinov3_analysis.json",analysis);add_json(z,"01_RESULTS/blind_adjudication.json",blind)
  for path in ["p586_plant_v01/results/blind_key.json","p586_plant_v01/results/blind_pair_candidates.json"]:
   try:add_json(z,"01_RESULTS/"+path.rsplit("/",1)[-1],jget(path))
   except Exception:pass
  for cid in IDS:
   pre=prefix(cid);whole=channels=None
   try:whole=jget(pre+"/whole_manifest_frozen.json");add_json(z,f"02_CORPORA/{cid}/whole_manifest_frozen.json",whole)
   except Exception:continue
   try:channels=jget(pre+"/channels_manifest_frozen.json");add_json(z,f"02_CORPORA/{cid}/channels_manifest_frozen.json",channels)
   except Exception:channels=None
   plants=(channels or whole).get("plants",[]);flat=[];repro=[];imgs=[];labs=[]
   for p in plants:
    flat.append({"plant_id":p.get("plant_id"),"qa_status":p.get("qa_status"),"qa_confidence":p.get("qa_confidence"),"mask_valid":p.get("mask_valid"),"mask_area_fraction":p.get("mask_area_fraction"),"root_boundary_y_1000":p.get("root_boundary_y_1000"),"crop_path":p.get("crop_path"),"masked_crop_path":p.get("masked_crop_path"),"above_strict_path":p.get("above_strict_path"),"above_context_path":p.get("above_context_path")})
    for zz in p.get("reproductive_structures",[]):repro.append({"plant_id":p.get("plant_id"),"repro_id":zz.get("repro_id"),"qa_class":zz.get("qa_class"),"qa_status":zz.get("qa_status"),"qa_confidence":zz.get("qa_confidence"),"crop_path":zz.get("crop_path")})
    for key,sub in [("crop_path","whole"),("masked_crop_path","masked"),("above_strict_path","above_strict"),("above_context_path","above_context")]:
     path=p.get(key)
     if path:
      try:
       data=get(BRIDGE+path).content;z.writestr(f"02_CORPORA/{cid}/images/{sub}/{path.rsplit('/',1)[-1]}",data)
       if key=="masked_crop_path" and len(imgs)<40:imgs.append(Image.open(io.BytesIO(data)).convert("RGB"));labs.append(f"{p.get('plant_id')} {p.get('qa_status')}")
      except Exception:pass
    for zz in p.get("reproductive_structures",[]):
     path=zz.get("crop_path")
     if path:
      try:z.writestr(f"02_CORPORA/{cid}/images/reproductive/{path.rsplit('/',1)[-1]}",get(BRIDGE+path).content)
      except Exception:pass
   if flat:z.writestr(f"02_CORPORA/{cid}/plants.csv",csv_bytes(flat,list(flat[0].keys())))
   if repro:z.writestr(f"02_CORPORA/{cid}/reproductive.csv",csv_bytes(repro,list(repro[0].keys())))
   if imgs:z.writestr(f"02_CORPORA/{cid}/masked_contact_sheet.jpg",contact(imgs,labs,f"{cid} masked whole plants (first 40)"))
   summary.append({"corpus":cid,"whole":len(plants),"valid_masks":sum(bool(x.get("mask_valid")) for x in plants),"reproductive":len(repro),"whole_manifest_sha256":whole.get("whole_manifest_sha256"),"channel_manifest_sha256":channels.get("channel_manifest_sha256") if channels else None})
  for t in blind.get("trials",[]):
   try:add_url(z,f"03_BLIND_SHEETS/{t['trial_id']}.png",BRIDGE+t["sheet_path"])
   except Exception:pass
  z.writestr("04_TABLES/corpus_summary.csv",csv_bytes(summary,list(summary[0].keys())))
  md=["# Palatino 586 ↔ Voynich Plant-Morphology Programme v0.1","",f"DINOv3 result SHA-256: `{analysis.get('result_sha256')}`",f"Blind result SHA-256: `{blind.get('result_sha256')}`","","## Corpus summary",""]
  for s in summary:md.append(f"- {s['corpus']}: {s['whole']} broad whole plants; {s['valid_masks']} valid masks; {s['reproductive']} reproductive proposals.")
  md += ["","## Primary channel results",""]
  for k,v in analysis.get("results",{}).items():
   if v.get("primary_effect_target_minus_control_mean") is not None:md.append(f"- {k}: effect {v.get('primary_effect_target_minus_control_mean'):.6f}; 95% CI {v.get('hierarchical_bootstrap_95ci')}; manuscript-rank p {v.get('manuscript_label_rank_p_upper')}; target rank {v.get('target_rank_descending')}.")
  md += ["","## Blind adjudication", "",json.dumps(blind.get("summary",{}),indent=2,sort_keys=True),"","The ZIP includes frozen protocols, manifests, ordinary/masked/above-ground/reproductive images, CSV ledgers, evidence sheets and corrected results. DINO vectors are preserved separately in the public experiment storage and identified by hashes in the analysis JSON."]
  z.writestr("README.md","\n".join(md))
 data=buf.getvalue();digest=hashlib.sha256(data).hexdigest();ticket=S.post(TICKET,headers={"x-upload-token":TOKEN},json={},timeout=120);ticket.raise_for_status();info=ticket.json();url=info.get("signedUrl") or info.get("signedURL");url=url if url.startswith("http") else SUPA+url;up=S.put(url,headers={"content-type":"application/zip","x-upsert":"true"},data=data,timeout=1800);up.raise_for_status();meta={"path":info["path"],"bytes":len(data),"sha256":digest,"analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256")};payload={"path":"p586_plant_v01/results/human_bundle.json","content_type":"application/json","data_b64":base64.b64encode(json.dumps(meta,indent=2,sort_keys=True).encode()).decode()};r=S.post(EP,headers={"x-upload-token":TOKEN},json=payload,timeout=120);r.raise_for_status();print("RESULT_JSON="+json.dumps(meta,sort_keys=True),flush=True)
if __name__=="__main__":main()
