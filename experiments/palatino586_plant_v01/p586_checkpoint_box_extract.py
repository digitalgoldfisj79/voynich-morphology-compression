#!/usr/bin/env python3
"""Freeze a preregistered whole-plant checkpoint control under experiment QA."""
from __future__ import annotations
import io,json,os,time,requests,torch
from PIL import Image,ImageOps
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/1f57a841e79f5677726025a2ecb70fe5901e6cd5/experiments/palatino586_plant_v01/p586_target_extract.py"
code=requests.get(BASE,timeout=120).text;ns={"__name__":"targetlib"};exec(compile(code,BASE,"exec"),ns)
get,upload,png,sha,csha,infer,qa,channel=[ns[k] for k in ("get","upload","png","sha","csha","infer","qa","channel")]
H=ns["H"];MODEL=ns["MODEL"]
CID=os.environ.get("CORPUS_ID","bnf_gr_2179");CORPUS=f"https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/p586-plant-corpus-v01?kind=plant_boxes&id={CID}";PREFIX=f"p586_plant_v01/controls/{CID}"
OBJECT_QA='''This is an existing whole-plant localisation crop from a medieval herbal corpus. Return strict JSON only: {"status":"accept|partial|reject|uncertain","confidence":0.0,"is_single_coherent_plant":true,"complete_fraction":0.0,"text_contamination":"none|minor|material","duplicate_or_overlap":false,"reason":"brief"}. Accept a complete/effectively complete coherent plant. Partial is genuine and morphologically useful but materially truncated or obscured. Reject non-plants, text, decoration, duplicate fragments or unusable crops.'''

def pad(b,w,h):
 x0,y0,x1,y1=b;bw=x1-x0;bh=y1-y0;return max(0,round(x0-.06*bw)),max(0,round(y0-.04*bh)),min(w,round(x1+.06*bw)),min(h,round(y1+.08*bh))
def bd(b):return {"x":b[0],"y":b[1],"w":b[2]-b[0],"h":b[3]-b[1]}
def save(out):
 out["counts"]={"source_boxes":out["source_boxes"],"processed":len(out["plants"]),**{k:sum(x["qa_status"]==k for x in out["plants"]) for k in ("accept","partial","reject","uncertain")}};out["counts"]["broad"]=out["counts"]["accept"]+out["counts"]["partial"];upload(PREFIX+"/checkpoint.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode())
def main():
 src=get(CORPUS).json();rows=src["rows"];P=H["AutoProcessor"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),min_pixels=256*28*28,max_pixels=1280*28*28);M=H["Qwen2_5_VLForConditionalGeneration"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval();rev=getattr(M.config,"_commit_hash",None) or "unknown"
 out={"protocol_id":"P586-VMS-PLANT-0.1-20260803","amendment":"PREREGISTRATION_AMENDMENT_02","manuscript_id":CID,"source":"existing herbal_plant_boxes localisation checkpoint, re-cropped and re-QA","source_boxes":len(rows),"broad_cap":20,"detector_model":MODEL,"detector_revision":rev,"plants":[]}
 page_cache={}
 for n,r in enumerate(rows,1):
  if out.get("counts",{}).get("broad",0)>=20:break
  fol=r.get("folio") or {};url=fol.get("image_url");slug=r.get("slug");pid=f"{slug}_b{n-1:03d}"
  try:
   if not url:raise RuntimeError("missing linked folio image URL")
   if url not in page_cache:
    raw=get(url).content;page_cache[url]=(ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB"),sha(raw))
   page,ph=page_cache[url];sx=page.width/max(1,int(r.get("img_w") or page.width));sy=page.height/max(1,int(r.get("img_h") or page.height));b=(round(float(r["x0"])*sx),round(float(r["y0"])*sy),round(float(r["x1"])*sx),round(float(r["y1"])*sy));cb=pad(b,page.width,page.height);crop=page.crop(cb);ans,rawans=infer(M,P,[crop],OBJECT_QA,450);st,conf,reason=qa(ans);data=png(crop);path=f"{PREFIX}/whole/{pid}.png";upload(path,"image/png",data)
   rec={"plant_id":pid,"manuscript_id":CID,"seq":fol.get("seq"),"slug":slug,"folio_canonical":fol.get("folio_canonical"),"source_image_url":url,"source_image_sha256":ph,"checkpoint_bbox":bd(b),"crop_bbox":bd(cb),"checkpoint_method":r.get("method"),"checkpoint_tight":r.get("tight"),"crop_path":path,"crop_sha256":sha(data),"qa_status":st,"qa_confidence":conf,"qa_reason":reason,"qa_raw":ans,"qa_response_raw":rawans,"reproductive_structures":[]}
   if st in {"accept","partial"}:
    ca,craw=infer(M,P,[crop],ns["CHANNEL"],750);roots,y,yc,yr,structs=channel(ca);rec.update(has_visible_roots=roots,root_boundary_y_1000=y,root_boundary_confidence=yc,root_boundary_reason=yr,channel_raw=ca,channel_response_raw=craw)
   else:rec.update(has_visible_roots=None,root_boundary_y_1000=None,root_boundary_confidence=None,root_boundary_reason="excluded by whole QA")
   out["plants"].append(rec)
  except Exception as e:out["plants"].append({"plant_id":pid,"manuscript_id":CID,"slug":slug,"qa_status":"reject","qa_confidence":0.0,"qa_reason":f"{type(e).__name__}: {e}","reproductive_structures":[]})
  save(out);print(json.dumps({"event":"checkpoint_box","corpus":CID,"n":n,"total":len(rows),"counts":out["counts"]},sort_keys=True),flush=True)
 out["whole_manifest_sha256"]=csha({k:out[k] for k in ("protocol_id","amendment","manuscript_id","source","source_boxes","broad_cap","detector_model","detector_revision","plants")});out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());save(out);upload(PREFIX+"/whole_manifest_frozen.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode());print("RESULT_JSON="+json.dumps({"manuscript_id":CID,"whole_manifest_sha256":out["whole_manifest_sha256"],"counts":out["counts"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
