#!/usr/bin/env python3
"""Freeze Voynich/BSB whole-plant crops under the same visual QA policy."""
from __future__ import annotations
import io,json,os,time,requests,torch
from PIL import Image
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/1f57a841e79f5677726025a2ecb70fe5901e6cd5/experiments/palatino586_plant_v01/p586_target_extract.py"
code=requests.get(BASE,timeout=120).text; ns={"__name__":"p586_target_lib"}; exec(compile(code,BASE,"exec"),ns)
get,upload,png,sha,csha,infer,channel=[ns[k] for k in ("get","upload","png","sha","csha","infer","channel")]
H=ns["H"]; MODEL=ns["MODEL"]
MID=os.environ["OBJECT_ID"]
CORPUS=f"https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/p586-plant-corpus-v01?kind=objects&id={MID}"
STORAGE="https://ymaqlcfjmdwncdbjprmw.supabase.co/storage/v1/object/public/manuscripts/"
PREFIX=f"p586_plant_v01/controls/{MID}"
OBJECT_QA='''This is an existing whole-plant crop from a medieval herbal corpus. Return strict JSON only: {"status":"accept|partial|reject|uncertain","confidence":0.0,"is_single_coherent_plant":true,"complete_fraction":0.0,"text_contamination":"none|minor|material","reason":"brief"}. Accept a complete/effectively complete coherent plant. Partial is morphologically useful but materially truncated or obscured. Reject non-plants, text, decoration, duplicate fragments or unusable crops.'''

def qa(a):
    if not isinstance(a,dict):return "uncertain",0.0,"malformed"
    st=str(a.get("status","uncertain")).lower(); cf=ns["num"](a.get("confidence")); comp=ns["num"](a.get("complete_fraction")); reason=str(a.get("reason",""))
    if st not in {"accept","partial","reject","uncertain"}:st="uncertain"
    if a.get("is_single_coherent_plant") is False:st="reject"
    if st=="accept" and (cf<.70 or comp<.70 or str(a.get("text_contamination","")).lower()=="material"):st="partial" if cf>=.50 else "uncertain"
    if st=="partial" and cf<.45:st="uncertain"
    return st,cf,reason

def save(out):
    out["counts"]={"source_objects":out["source_objects"],"processed":len(out["plants"]),**{k:sum(x["qa_status"]==k for x in out["plants"]) for k in ("accept","partial","reject","uncertain")}}
    out["counts"]["broad"]=out["counts"]["accept"]+out["counts"]["partial"]
    upload(PREFIX+"/checkpoint.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode())

def main():
    source=get(CORPUS).json(); rows=source["rows"]
    P=H["AutoProcessor"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),min_pixels=256*28*28,max_pixels=1280*28*28)
    M=H["Qwen2_5_VLForConditionalGeneration"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval(); rev=getattr(M.config,"_commit_hash",None) or "unknown"
    cap=None if MID=="voynich" else 20
    out={"protocol_id":"P586-VMS-PLANT-0.1-20260803","manuscript_id":MID,"source":"existing Plant Mask Painter whole crops, re-QA under experiment policy","source_objects":len(rows),"broad_cap":cap,"detector_model":MODEL,"detector_revision":rev,"plants":[]}
    for i,r in enumerate(rows,1):
        if cap is not None and out.get("counts",{}).get("broad",0)>=cap:break
        try:
            url=STORAGE+r["crop_path"]; raw=get(url).content; im=Image.open(io.BytesIO(raw)).convert("RGB"); ans,araw=infer(M,P,[im],OBJECT_QA,420); st,conf,reason=qa(ans); pid=f"{MID}_s{int(r['seq']):04d}_o{int(r['obj_index']):02d}"; data=png(im); path=f"{PREFIX}/whole/{pid}.png"; upload(path,"image/png",data)
            rec={"plant_id":pid,"source_object_id":r["id"],"manuscript_id":MID,"seq":r["seq"],"slug":r["slug"],"obj_index":r["obj_index"],"source_crop_path":r["crop_path"],"source_crop_url":url,"source_bbox":r.get("bbox"),"crop_path":path,"crop_sha256":sha(data),"qa_status":st,"qa_confidence":conf,"qa_reason":reason,"qa_raw":ans,"qa_response_raw":araw,"reproductive_structures":[]}
            if st in {"accept","partial"}:
                ca,craw=infer(M,P,[im],ns["CHANNEL"],750); roots,y,yc,yr,structs=channel(ca); rec.update(has_visible_roots=roots,root_boundary_y_1000=y,root_boundary_confidence=yc,root_boundary_reason=yr,channel_raw=ca,channel_response_raw=craw)
            else:rec.update(has_visible_roots=None,root_boundary_y_1000=None,root_boundary_confidence=None,root_boundary_reason="excluded by whole QA")
            out["plants"].append(rec)
        except Exception as e: out["plants"].append({"plant_id":f"error_{i}","manuscript_id":MID,"source_object_id":r.get("id"),"qa_status":"reject","qa_confidence":0.0,"qa_reason":f"{type(e).__name__}: {e}","reproductive_structures":[]})
        save(out); print(json.dumps({"event":"existing_object","manuscript":MID,"n":i,"total":len(rows),"counts":out["counts"]},sort_keys=True),flush=True)
    out["whole_manifest_sha256"]=csha({k:out[k] for k in ("protocol_id","manuscript_id","source","source_objects","broad_cap","detector_model","detector_revision","plants")}); out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); save(out); upload(PREFIX+"/whole_manifest_frozen.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode()); print("RESULT_JSON="+json.dumps({"manuscript_id":MID,"whole_manifest_sha256":out["whole_manifest_sha256"],"counts":out["counts"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
