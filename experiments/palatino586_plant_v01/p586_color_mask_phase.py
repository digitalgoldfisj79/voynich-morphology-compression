#!/usr/bin/env python3
"""Frozen Plant Mask Painter colour-mask phase; no DINO or similarity is computed."""
from __future__ import annotations
import base64, concurrent.futures, hashlib, io, json, os, re, threading, time
from pathlib import Path
from typing import Any
import numpy as np, requests
from PIL import Image

PROTOCOL="P586-VMS-PLANT-0.1-20260803"
PROMPT_SHA="bc834fb40c1afb10dc11ca7e9f1a979836e3bc3146642b84c71a078552c7a38e"
MODEL="google/gemini-3.1-flash-image"
TARGET="bncf_palatino_586"
CID=os.environ["CORPUS_ID"]
WORKERS=max(1,min(6,int(os.environ.get("MASK_WORKERS","3"))))
SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co"
BRIDGE=SUPA+"/storage/v1/object/public/bridge/"
UPLOAD_EP=SUPA+"/functions/v1/p586-plant-upload-v01"
MASK_EP="https://plant-mask-painter.lovable.app/api/p586-mask-v01"
ROOT_SOURCE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py"
PREFIX="p586_plant_v01/target" if CID==TARGET else f"p586_plant_v01/controls/{CID}"
S=requests.Session(); S.headers["User-Agent"]="P586ColourMask/0.1"
LOCK=threading.Lock()


def get(url:str,timeout=180):
    r=S.get(url,timeout=timeout); r.raise_for_status(); return r

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def csha(x:Any)->str:return sha(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode())
def token()->str:
    text=get(ROOT_SOURCE,60).text
    value=re.search(r'RUN_ID\s*=\s*"([^"]+)"',text).group(1)
    if sha(value.encode())!="116954ffdc209c006292b4c5dcc96cbd0eddeb3e1c7ea788aeabb68b6855a929":raise RuntimeError("token hash mismatch")
    return value
TOKEN=token()

def upload(path:str,typ:str,data:bytes):
    payload={"path":path,"content_type":typ,"data_b64":base64.b64encode(data).decode()};last=None
    for k in range(5):
        try:
            r=requests.post(UPLOAD_EP,headers={"x-upload-token":TOKEN},json=payload,timeout=360);r.raise_for_status();return
        except Exception as e:last=e;time.sleep(min(15,1.7**k))
    raise RuntimeError(f"upload {path}: {last}")
def png(im:Image.Image)->bytes:
    b=io.BytesIO();im.save(b,"PNG",optimize=True);return b.getvalue()

def load_checkpoint(whole_sha:str)->dict:
    url=BRIDGE+PREFIX+"/color_mask_checkpoint.json"
    try:
        x=get(url,120).json()
        if x.get("protocol_id")!=PROTOCOL or x.get("corpus_id")!=CID or x.get("whole_manifest_sha256")!=whole_sha:raise RuntimeError("checkpoint identity mismatch")
        return x
    except Exception:
        return {"protocol_id":PROTOCOL,"corpus_id":CID,"whole_manifest_sha256":whole_sha,"model":MODEL,"prompt_sha256":PROMPT_SHA,"implementation_freeze_commit":"1a26e38bd70fa3b072b8cdf64775d712573e5f39","records":[]}

def save(out:dict):
    rec=out["records"]
    out["counts"]={"records":len(rec),"success":sum(x.get("status")=="success" for x in rec),"failed":sum(x.get("status")!="success" for x in rec),"area_valid":sum(x.get("area_valid",False) for x in rec),"area_invalid":sum(x.get("status")=="success" and not x.get("area_valid",False) for x in rec),"above_valid":sum(x.get("above_valid",False) for x in rec),"resized":sum(x.get("mask_resized",False) for x in rec)}
    upload(PREFIX+"/color_mask_checkpoint.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode())

def endpoint(source_url:str)->dict:
    last=None
    for k in range(4):
        try:
            r=requests.post(MASK_EP,headers={"x-p586-token":TOKEN},json={"imageUrl":source_url},timeout=420)
            if r.status_code in {429,500,502,503,504}:raise RuntimeError(f"endpoint transient {r.status_code}: {r.text[:200]}")
            r.raise_for_status();return r.json()
        except Exception as e:last=e;time.sleep(min(30,2**k))
    raise RuntimeError(f"mask endpoint failed: {last}")

def convert(plant:dict)->dict:
    pid=plant["plant_id"];source_path=plant["crop_path"];source_url=BRIDGE+source_path
    try:
        source_bytes=get(source_url,180).content;source_sha=sha(source_bytes);source=Image.open(io.BytesIO(source_bytes)).convert("RGB")
        response=endpoint(source_url)
        if response.get("model")!=MODEL:raise RuntimeError(f"model mismatch: {response.get('model')}")
        if response.get("promptSha256")!=PROMPT_SHA:raise RuntimeError(f"prompt hash mismatch: {response.get('promptSha256')}")
        if response.get("sourceSha256")!=source_sha:raise RuntimeError("endpoint/local source hash mismatch")
        raw=base64.b64decode(response["pngBase64"]);raw_im=Image.open(io.BytesIO(raw)).convert("RGB");returned_size=list(raw_im.size);resized=raw_im.size!=source.size
        if resized:raw_im=raw_im.resize(source.size,Image.Resampling.NEAREST)
        a=np.asarray(raw_im,dtype=np.int16)
        colors=np.array([[0,255,0],[255,0,0],[255,255,255]],dtype=np.int16)
        dist=((a[:,:,None,:]-colors[None,None,:,:])**2).sum(axis=3)
        cls=np.argmin(dist,axis=2).astype(np.uint8)
        q=colors[cls].astype(np.uint8);quant=Image.fromarray(q,"RGB")
        green=cls==0;red=cls==1;fg=green|red
        total=cls.size;gf=float(green.sum()/total);rf=float(red.sum()/total);wf=float((cls==2).sum()/total);ff=gf+rf
        source_arr=np.asarray(source,dtype=np.uint8);white=np.full_like(source_arr,255)
        masked=white.copy();masked[fg]=source_arr[fg]
        above=white.copy();above[green]=source_arr[green]
        if green.any():
            low=int(np.where(green)[0].max());limit=min(source.height-1,low+round(source.height*.05));context_mask=fg & (np.arange(source.height)[:,None]<=limit);context=white.copy();context[context_mask]=source_arr[context_mask];above_valid=ff>=.01 and ff<=.90 and gf>=.005
        else:
            low=None;limit=None;context=white.copy();above_valid=False
        area_valid=.01<=ff<=.90
        raw_path=f"{PREFIX}/color_masks/raw/{pid}.png";quant_path=f"{PREFIX}/color_masks/quantized/{pid}.png";masked_path=f"{PREFIX}/masked/{pid}.png";above_path=f"{PREFIX}/above_strict/{pid}.png";context_path=f"{PREFIX}/above_context/{pid}.png"
        qbytes=png(quant);mbytes=png(Image.fromarray(masked));abytes=png(Image.fromarray(above));cbytes=png(Image.fromarray(context))
        upload(raw_path,"image/png",raw);upload(quant_path,"image/png",qbytes);upload(masked_path,"image/png",mbytes);upload(above_path,"image/png",abytes);upload(context_path,"image/png",cbytes)
        nearest=np.sqrt(np.min(dist,axis=2).astype(np.float64))
        return {"plant_id":pid,"qa_status":plant.get("qa_status"),"status":"success","source_crop_path":source_path,"source_sha256":source_sha,"source_size":list(source.size),"model":response.get("model"),"prompt_sha256":response.get("promptSha256"),"response_id":response.get("responseId"),"response_created":response.get("created"),"raw_mask_path":raw_path,"raw_mask_sha256":sha(raw),"returned_mask_size":returned_size,"mask_resized":resized,"quantized_mask_path":quant_path,"quantized_mask_sha256":sha(qbytes),"green_fraction":gf,"red_fraction":rf,"white_fraction":wf,"foreground_fraction":ff,"mean_nearest_colour_distance":float(nearest.mean()),"p95_nearest_colour_distance":float(np.quantile(nearest,.95)),"area_valid":area_valid,"above_valid":above_valid,"lowest_green_row":low,"context_limit_row":limit,"masked_crop_path":masked_path,"masked_crop_sha256":sha(mbytes),"above_strict_path":above_path,"above_strict_sha256":sha(abytes),"above_context_path":context_path,"above_context_sha256":sha(cbytes)}
    except Exception as e:
        return {"plant_id":pid,"qa_status":plant.get("qa_status"),"status":"failed","source_crop_path":source_path,"error":f"{type(e).__name__}: {e}","area_valid":False,"above_valid":False}

def main():
    whole=get(BRIDGE+PREFIX+"/whole_manifest_frozen.json",180).json();whole_sha=whole["whole_manifest_sha256"]
    plants=[x for x in whole["plants"] if x.get("qa_status") in {"accept","partial"}]
    out=load_checkpoint(whole_sha);existing={x["plant_id"]:x for x in out["records"]};todo=[p for p in plants if existing.get(p["plant_id"],{}).get("status")!="success"]
    print(json.dumps({"event":"start","corpus":CID,"broad":len(plants),"already_success":len(plants)-len(todo),"todo":len(todo),"workers":WORKERS},sort_keys=True),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        fut={pool.submit(convert,p):p for p in todo}
        for n,f in enumerate(concurrent.futures.as_completed(fut),1):
            rec=f.result();existing[rec["plant_id"]]=rec
            with LOCK:
                out["records"]=[existing[p["plant_id"]] for p in plants if p["plant_id"] in existing];save(out)
            print(json.dumps({"event":"mask_complete","corpus":CID,"n":n,"todo":len(todo),"plant":rec["plant_id"],"status":rec["status"],"area_valid":rec.get("area_valid"),"foreground":round(rec.get("foreground_fraction",0),4)},sort_keys=True),flush=True)
    out["records"]=[existing[p["plant_id"]] for p in plants if p["plant_id"] in existing]
    out["complete"]=(len(out["records"])==len(plants) and all(x.get("status")=="success" for x in out["records"]));out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());out["color_mask_manifest_sha256"]=csha({k:out[k] for k in ("protocol_id","corpus_id","whole_manifest_sha256","model","prompt_sha256","implementation_freeze_commit","records")});save(out);upload(PREFIX+"/color_masks_frozen.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode());print("RESULT_JSON="+json.dumps({"corpus_id":CID,"complete":out["complete"],"color_mask_manifest_sha256":out["color_mask_manifest_sha256"],"counts":out["counts"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
