#!/usr/bin/env python3
"""Sealed fresh Palatino 586 whole-plant extraction; no similarity is computed."""
from __future__ import annotations
import base64, gc, hashlib, io, json, math, os, re, time
from pathlib import Path
from typing import Any
import requests, torch
from PIL import Image, ImageDraw, ImageOps

PROTOCOL="P586-VMS-PLANT-0.1-20260803"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co"
BRIDGE=SUPA+"/storage/v1/object/public/bridge/"
ROOT_CHECKPOINT=BRIDGE+"p586_root_v01/full_run/checkpoint_recovered.json"
ROOT_SOURCE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py"
UPLOAD_EP=SUPA+"/functions/v1/p586-plant-upload-v01"
PREFIX="p586_plant_v01/target"
S=requests.Session(); S.headers["User-Agent"]="P586PlantMorphology/0.1"

DETECT='''Return strict JSON only. Identify EVERY distinct coherent botanical plant illustration on this medieval herbal page, including plants without roots and multiple plants per page. Exclude handwriting, initials, borders, people, animals, stains, decoration and loose fragments. Schema: {"plants":[{"plant_index":0,"bbox_1000":[x0,y0,x1,y1],"complete":"complete|partial|uncertain","has_visible_root":true,"confidence":0.0,"description":"brief"}]}. Coordinates are normalized 0-1000 and cover each plant from topmost foliage/flower to lowest root or stem base.'''
QA='''Image 1 is the folio with one numbered red box; Image 2 is its generous crop. Return strict JSON only: {"status":"accept|partial|reject|uncertain","confidence":0.0,"is_single_coherent_plant":true,"complete_fraction":0.0,"text_contamination":"none|minor|material","duplicate_or_overlap":false,"reason":"brief"}. Accept a complete/effectively complete coherent plant. Partial is genuine and useful but materially truncated/obscured. Reject non-plants, text, decoration, fragments, duplicates or unusable crops.'''
CHANNEL='''Return strict JSON only for this adjudicated medieval plant crop: {"has_visible_roots":true,"root_boundary_y_1000":750,"root_boundary_confidence":0.0,"root_boundary_reason":"brief","reproductive_structures":[{"proposal_index":0,"class":"flower|flower_head|inflorescence|bud|fruit|seed_head","bbox_1000":[x0,y0,x1,y1],"confidence":0.0,"description":"brief"}]}. The root boundary is the first row below which coherent root/bulb/rhizome begins; use 1000 if no root. At most five structures; exclude leaves, roots, text and decoration.'''
RQA='''Image 1 is the complete plant; Image 2 is a proposed reproductive structure. Return strict JSON only: {"status":"accept|partial|reject|uncertain","class":"flower|flower_head|inflorescence|bud|fruit|seed_head|none","confidence":0.0,"attached_to_plant":true,"reason":"brief"}. Reject leaves, roots, text, decoration and stains.'''

def get(url, tries=5):
    last=None
    for k in range(tries):
        try:
            r=S.get(url,timeout=180); r.raise_for_status(); return r
        except Exception as e: last=e; time.sleep(min(15,1.7**k))
    raise RuntimeError(f"GET {url}: {last}")

def load_root_helpers():
    src=get(ROOT_SOURCE).text; ns={"__name__":"p586_root_helpers"}; exec(compile(src,ROOT_SOURCE,"exec"),ns); return ns,src

H,ROOT_TEXT=load_root_helpers()
TOKEN=re.search(r'RUN_ID\s*=\s*"([^"]+)"',ROOT_TEXT).group(1)

def upload(path,typ,data):
    payload={"path":path,"content_type":typ,"data_b64":base64.b64encode(data).decode()}
    last=None
    for k in range(5):
        try:
            r=S.post(UPLOAD_EP,headers={"x-upload-token":TOKEN},json=payload,timeout=300); r.raise_for_status(); return
        except Exception as e: last=e; time.sleep(min(15,1.8**k))
    raise RuntimeError(f"UPLOAD {path}: {last}")

def sha(b): return hashlib.sha256(b).hexdigest()
def csha(x): return sha(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode())
def png(im):
    b=io.BytesIO(); im.save(b,"PNG",optimize=True); return b.getvalue()
def jpg(im):
    b=io.BytesIO(); im.convert("RGB").save(b,"JPEG",quality=88,optimize=True); return b.getvalue()
def num(x,d=0.0):
    try: return float(x)
    except Exception: return d

def box(v,w,h,minpx=30): return H["normalise_box"](v,w,h,minpx)
def iou(a,b): return H["intersection_over_union"](a,b)
def bd(b): return {"x":b[0],"y":b[1],"w":b[2]-b[0],"h":b[3]-b[1]}
def expand(b,w,h,l=.06,r=.06,t=.04,bot=.08):
    x0,y0,x1,y1=b; bw=x1-x0; bh=y1-y0
    return max(0,round(x0-l*bw)),max(0,round(y0-t*bh)),min(w,round(x1+r*bw)),min(h,round(y1+bot*bh))

def infer(model,proc,images,prompt,tokens): return H["infer"](model,proc,images,prompt,tokens)
def plants(ans): return H["plants_from"](ans)

def qa(ans):
    if not isinstance(ans,dict): return "uncertain",0.0,"malformed"
    st=str(ans.get("status","uncertain")).lower(); cf=num(ans.get("confidence")); complete=num(ans.get("complete_fraction")); reason=str(ans.get("reason",""))
    if st not in {"accept","partial","reject","uncertain"}: st="uncertain"
    if ans.get("duplicate_or_overlap") or ans.get("is_single_coherent_plant") is False: st="reject"
    if st=="accept" and (cf<.70 or complete<.70 or str(ans.get("text_contamination","")).lower()=="material"): st="partial" if cf>=.50 else "uncertain"
    if st=="partial" and cf<.45: st="uncertain"
    return st,cf,reason

def channel(ans):
    if not isinstance(ans,dict): return False,1000,0.0,"malformed",[]
    roots=bool(ans.get("has_visible_roots")); y=round(num(ans.get("root_boundary_y_1000"),1000)); y=max(150,min(1000,y)) if roots else 1000
    allowed={"flower","flower_head","inflorescence","bud","fruit","seed_head"}; out=[]
    for i,z in enumerate(ans.get("reproductive_structures",[])[:5] if isinstance(ans.get("reproductive_structures"),list) else []):
        if isinstance(z,dict) and str(z.get("class","")).lower() in allowed: out.append({**z,"proposal_index":i,"class":str(z["class"]).lower()})
    return roots,y,num(ans.get("root_boundary_confidence")),str(ans.get("root_boundary_reason","")),out

def rqa(ans):
    allowed={"flower","flower_head","inflorescence","bud","fruit","seed_head"}
    if not isinstance(ans,dict): return "uncertain","none",0.0,"malformed"
    st=str(ans.get("status","uncertain")).lower(); cls=str(ans.get("class","none")).lower(); cf=num(ans.get("confidence")); reason=str(ans.get("reason",""))
    if st not in {"accept","partial","reject","uncertain"}: st="uncertain"
    if cls not in allowed or ans.get("attached_to_plant") is False: st="reject"; cls="none"
    if st=="accept" and cf<.70: st="partial" if cf>=.50 else "uncertain"
    return st,cls,cf,reason

def marked(page,b,label):
    im=page.copy(); d=ImageDraw.Draw(im); d.rectangle(b,outline="red",width=max(3,max(page.size)//500)); d.text((b[0]+3,max(0,b[1]-22)),label,fill="red"); return im

def save(out):
    out["counts"]={k:sum(p["qa_status"]==k for p in out["plants"]) for k in ("accept","partial","reject","uncertain")}
    out["counts"].update(pages=len(out["pages"]),whole_proposals=len(out["plants"]),reproductive_proposals=sum(len(p.get("reproductive_structures",[])) for p in out["plants"]))
    upload(PREFIX+"/checkpoint.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode())

def sheet(rows,crops,title):
    cols=4; cw,ch=300,350; nr=max(1,math.ceil(len(rows)/cols)); im=Image.new("RGB",(cols*cw,45+nr*ch),"white"); d=ImageDraw.Draw(im); d.text((8,10),title,fill="black")
    for i,r in enumerate(rows):
        y=45+(i//cols)*ch; x=(i%cols)*cw; q=ImageOps.contain(crops[r["plant_id"]],(cw-16,ch-58)); im.paste(q,(x+(cw-q.width)//2,y)); d.text((x+6,y+ch-48),f'{r["plant_id"]} {r["qa_status"]} {r["qa_confidence"]:.2f}',fill="black"); d.text((x+6,y+ch-28),f'rootY={r.get("root_boundary_y_1000")} repro={len(r.get("reproductive_structures",[]))}',fill="black")
    return im

def main():
    root=get(ROOT_CHECKPOINT).json(); src=sorted(root.get("pages",[]),key=lambda x:int(x["canvas_index"])); ids=[int(x["canvas_index"]) for x in src]
    if len(src)!=66 or len(set(ids))!=66: raise RuntimeError(f"expected 66 distinct source pages, got {len(src)}/{len(set(ids))}")
    freeze=[{k:p.get(k) for k in ("canvas_index","canvas_label","source_image_url","image_width","image_height","sha256")} for p in src]; fsha=csha(freeze)
    upload(PREFIX+"/source_freeze.json","application/json",json.dumps({"protocol_id":PROTOCOL,"root_result_sha256":root.get("result_sha256"),"pages":freeze,"source_freeze_sha256":fsha},indent=2,sort_keys=True).encode())
    P=H["AutoProcessor"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),min_pixels=256*28*28,max_pixels=1280*28*28)
    M=H["Qwen2_5_VLForConditionalGeneration"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval(); rev=getattr(M.config,"_commit_hash",None) or "unknown"
    out={"protocol_id":PROTOCOL,"target_manuscript_id":"bncf_palatino_586","source_relation":"exact frozen 66-page root-run set; no root-conditioned inclusion","source_freeze_sha256":fsha,"root_result_sha256":root.get("result_sha256"),"detector_model":MODEL,"detector_revision":rev,"detector_preprocessing":{"min_pixels":256*28*28,"max_pixels":1280*28*28,"decoding":"greedy"},"pages":[],"plants":[]}; crops={}
    for n,p in enumerate(src,1):
        idx=int(p["canvas_index"]); pr={"canvas_index":idx,"canvas_label":p.get("canvas_label"),"source_image_url":p.get("source_image_url"),"status":"pending"}
        try:
            raw=get(p["source_image_url"]).content; page=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB"); ph=sha(raw)
            ans,rawans=infer(M,P,[page],DETECT,900); cand=plants(ans); parsed=[]
            for pos,z in enumerate(cand):
                b=box(z.get("bbox_1000") or z.get("bbox"),page.width,page.height,55) if isinstance(z,dict) else None
                if b: parsed.append((pos,z,b,num(z.get("confidence"))))
            parsed.sort(key=lambda q:(-q[3],q[0])); keep=[]
            for q in parsed:
                if not any(iou(q[2],x[2])>.65 for x in keep): keep.append(q)
            keep.sort(key=lambda q:(q[2][1],q[2][0],q[0])); pr.update(status="processed",image_width=page.width,image_height=page.height,source_image_sha256=ph,source_hash_matches_root=(not p.get("sha256") or p.get("sha256")==ph),raw_candidates=len(cand),valid_candidates=len(parsed),unique_candidates=len(keep),detector_raw=rawans)
            for j,(pos,z,b,dc) in enumerate(keep):
                cb=expand(b,page.width,page.height); crop=page.crop(cb); pid=f"c{idx:03d}_p{j:02d}"; qans,qraw=infer(M,P,[marked(page,cb,pid),crop],QA,550); st,qc,reason=qa(qans); data=png(crop); path=f"{PREFIX}/whole/{pid}.png"; upload(path,"image/png",data); crops[pid]=crop
                rec={"plant_id":pid,"manuscript_id":"bncf_palatino_586","canvas_index":idx,"canvas_label":p.get("canvas_label"),"source_image_url":p.get("source_image_url"),"plant_index":j,"detector_position":pos,"detector_confidence":dc,"detector_complete":z.get("complete"),"detector_has_visible_root":z.get("has_visible_root"),"detector_description":z.get("description"),"bbox":bd(b),"crop_bbox":bd(cb),"crop_path":path,"crop_sha256":sha(data),"qa_status":st,"qa_confidence":qc,"qa_reason":reason,"qa_raw":qans,"qa_response_raw":qraw,"reproductive_structures":[]}
                if st in {"accept","partial"}:
                    cans,craw=infer(M,P,[crop],CHANNEL,750); roots,y,yc,yr,structs=channel(cans); rec.update(has_visible_roots=roots,root_boundary_y_1000=y,root_boundary_confidence=yc,root_boundary_reason=yr,channel_raw=cans,channel_response_raw=craw)
                    for z2 in structs:
                        rb=box(z2.get("bbox_1000") or z2.get("bbox"),crop.width,crop.height,14)
                        if not rb: continue
                        rcb=expand(rb,crop.width,crop.height,.12,.12,.12,.12); rim=crop.crop(rcb); a,araw=infer(M,P,[crop,rim],RQA,450); rs,rc,rconf,rr=rqa(a); rid=f"{pid}_r{len(rec['reproductive_structures']):02d}"; rdata=png(rim); rpath=f"{PREFIX}/reproductive/{rid}.png"; upload(rpath,"image/png",rdata)
                        rec["reproductive_structures"].append({"repro_id":rid,"proposal_index":z2.get("proposal_index"),"proposed_class":z2.get("class"),"proposal_confidence":num(z2.get("confidence")),"proposal_description":z2.get("description"),"bbox":bd(rb),"crop_bbox":bd(rcb),"crop_path":rpath,"crop_sha256":sha(rdata),"qa_status":rs,"qa_class":rc,"qa_confidence":rconf,"qa_reason":rr,"qa_raw":a,"qa_response_raw":araw})
                else: rec.update(has_visible_roots=None,root_boundary_y_1000=None,root_boundary_confidence=None,root_boundary_reason="excluded by whole-plant QA")
                out["plants"].append(rec)
            out["pages"].append(pr)
        except Exception as e: pr.update(status="error",error=f"{type(e).__name__}: {e}"); out["pages"].append(pr)
        save(out); print(json.dumps({"event":"page_complete","page":n,"total":66,"canvas":idx,"counts":out["counts"]},sort_keys=True),flush=True)
    out["pages"].sort(key=lambda x:int(x["canvas_index"])); out["plants"].sort(key=lambda x:(int(x["canvas_index"]),int(x["plant_index"])))
    payload={k:out[k] for k in ("protocol_id","target_manuscript_id","source_freeze_sha256","root_result_sha256","detector_model","detector_revision","detector_preprocessing","pages","plants")}; out["whole_manifest_sha256"]=csha(payload); out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); save(out)
    upload(PREFIX+"/whole_manifest_frozen.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode())
    groups={"all":out["plants"],"accept":[x for x in out["plants"] if x["qa_status"]=="accept"],"broad":[x for x in out["plants"] if x["qa_status"] in {"accept","partial"}],"reject_uncertain":[x for x in out["plants"] if x["qa_status"] in {"reject","uncertain"}]}
    for name,rows in groups.items(): upload(f"{PREFIX}/contact_sheets/{name}.jpg","image/jpeg",jpg(sheet(rows,crops,f"Palatino 586 whole plants — {name}")))
    report={"protocol_id":PROTOCOL,"source_pages":66,"source_freeze_sha256":fsha,"whole_manifest_sha256":out["whole_manifest_sha256"],"model":MODEL,"model_revision":rev,"counts":out["counts"],"errors":[p for p in out["pages"] if p["status"]=="error"],"no_similarity_computed":True}; upload(PREFIX+"/extraction_report.json","application/json",json.dumps(report,indent=2,sort_keys=True).encode()); print("RESULT_JSON="+json.dumps(report,sort_keys=True),flush=True)
    del M,P; gc.collect(); torch.cuda.empty_cache()

if __name__=="__main__": main()
