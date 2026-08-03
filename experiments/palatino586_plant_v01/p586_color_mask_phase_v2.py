#!/usr/bin/env python3
"""Audited correction of RGB-distance overflow; reuses every successful model output.

The v0.1 converter squared int16 colour differences, which can overflow above 181
and corrupt nearest-colour assignment. This launcher switches only that arithmetic
to int32 and reprocesses already stored raw model PNGs without regenerating them.
"""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/13b00353fe673fb3e44c0eadb31ee18550a5a8fd/experiments/palatino586_plant_v01/p586_color_mask_phase.py"
source=requests.get(URL,timeout=120).text
source=source.replace('LOCK=threading.Lock()\n','LOCK=threading.Lock()\nEXISTING_RECORDS={}\n',1)
old='''        response=endpoint(source_url)\n        if response.get("model")!=MODEL:raise RuntimeError(f"model mismatch: {response.get('model')}")\n        if response.get("promptSha256")!=PROMPT_SHA:raise RuntimeError(f"prompt hash mismatch: {response.get('promptSha256')}")\n        if response.get("sourceSha256")!=source_sha:raise RuntimeError("endpoint/local source hash mismatch")\n        raw=base64.b64decode(response["pngBase64"]);raw_im=Image.open(io.BytesIO(raw)).convert("RGB");returned_size=list(raw_im.size);resized=raw_im.size!=source.size'''
new='''        prior=EXISTING_RECORDS.get(pid,{})\n        if prior.get("status")=="success" and prior.get("raw_mask_path"):\n            raw=get(BRIDGE+prior["raw_mask_path"],180).content\n            response={"model":prior.get("model"),"promptSha256":prior.get("prompt_sha256"),"sourceSha256":prior.get("source_sha256"),"responseId":prior.get("response_id"),"created":prior.get("response_created")}\n            model_output_reused=True\n        else:\n            response=endpoint(source_url)\n            raw=base64.b64decode(response["pngBase64"])\n            model_output_reused=False\n        if response.get("model")!=MODEL:raise RuntimeError(f"model mismatch: {response.get('model')}")\n        if response.get("promptSha256")!=PROMPT_SHA:raise RuntimeError(f"prompt hash mismatch: {response.get('promptSha256')}")\n        if response.get("sourceSha256")!=source_sha:raise RuntimeError("endpoint/local source hash mismatch")\n        raw_im=Image.open(io.BytesIO(raw)).convert("RGB");returned_size=list(raw_im.size);resized=raw_im.size!=source.size'''
if source.count(old)!=1:raise RuntimeError(f"reuse patch expected once, found {source.count(old)}")
source=source.replace(old,new,1)
source=source.replace('a=np.asarray(raw_im,dtype=np.int16)','a=np.asarray(raw_im,dtype=np.int32)',1)
source=source.replace('dtype=np.int16)\n        dist=', 'dtype=np.int32)\n        dist=',1)
oldret='''return {"plant_id":pid,"qa_status":plant.get("qa_status"),"status":"success","source_crop_path":source_path,'''
newret='''return {"plant_id":pid,"qa_status":plant.get("qa_status"),"status":"success","conversion_version":"rgb-nearest-v2-int32","model_output_reused":model_output_reused,"source_crop_path":source_path,'''
if source.count(oldret)!=1:raise RuntimeError("return patch not found")
source=source.replace(oldret,newret,1)
oldmain='''    out=load_checkpoint(whole_sha);existing={x["plant_id"]:x for x in out["records"]};todo=[p for p in plants if existing.get(p["plant_id"],{}).get("status")!="success"]'''
newmain='''    out=load_checkpoint(whole_sha);existing={x["plant_id"]:x for x in out["records"]}\n    global EXISTING_RECORDS;EXISTING_RECORDS=existing\n    todo=[p for p in plants if existing.get(p["plant_id"],{}).get("conversion_version")!="rgb-nearest-v2-int32"]'''
if source.count(oldmain)!=1:raise RuntimeError("main patch not found")
source=source.replace(oldmain,newmain,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
