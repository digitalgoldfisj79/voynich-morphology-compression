#!/usr/bin/env python3
"""SAM2 masks, root-excluded crops and mask-based reproductive extraction."""
from __future__ import annotations
import gc, io, json, os, time, urllib.request
import cv2, numpy as np, requests, torch
from PIL import Image, ImageOps
from transformers import Sam2Model, Sam2Processor

TARGET_SOURCE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/1f57a841e79f5677726025a2ecb70fe5901e6cd5/experiments/palatino586_plant_v01/p586_target_extract.py"
ns={"__name__":"p586_target_module"}; exec(compile(requests.get(TARGET_SOURCE,timeout=120).text,TARGET_SOURCE,"exec"),ns)
get,upload,png,jpg,sha,csha,box,expand,bd,infer,rqa=ns["get"],ns["upload"],ns["png"],ns["jpg"],ns["sha"],ns["csha"],ns["box"],ns["expand"],ns["bd"],ns["infer"],ns["rqa"]
H=ns["H"]; MODEL=ns["MODEL"]; PREFIX="p586_plant_v01/target"; BRIDGE=ns["BRIDGE"]
SAM_ID="facebook/sam2.1-hiera-large"
MASKED_REPRO='''This is a white-background masked medieval plant. Return strict JSON only: {"reproductive_structures":[{"proposal_index":0,"class":"flower|flower_head|inflorescence|bud|fruit|seed_head","bbox_1000":[x0,y0,x1,y1],"confidence":0.0,"description":"brief"}]}. Propose at most five coherent reproductive structures. Exclude ordinary leaves, stems, roots, text and background.'''

def clean_mask(mask, relbox):
    m=(mask>0).astype(np.uint8); h,w=m.shape; n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
    keep=np.zeros_like(m); bx0,by0,bx1,by1=relbox
    for i in range(1,n):
        x,y,cw,ch,area=stats[i]
        inter=max(0,min(x+cw,bx1)-max(x,bx0))*max(0,min(y+ch,by1)-max(y,by0))
        if area>=max(12,int(h*w*.00015)) and (inter>0 or area>=h*w*.002): keep[lab==i]=1
    k=max(3,(min(h,w)//120)|1); kernel=np.ones((k,k),np.uint8); keep=cv2.morphologyEx(keep,cv2.MORPH_CLOSE,kernel)
    return keep.astype(bool)

def select_mask(outputs,processor,inputs):
    ms=processor.post_process_masks(outputs.pred_masks.detach().cpu(),inputs["original_sizes"].detach().cpu())[0]
    scores=outputs.iou_scores.detach().cpu().reshape(-1)
    if ms.ndim==4: ms=ms[0]
    if ms.ndim==2: return ms.numpy()>0,0,float(scores[0])
    j=int(torch.argmax(scores[:ms.shape[0]])); return ms[j].numpy()>0,j,float(scores[j])

def main():
    manifest=get(BRIDGE+PREFIX+"/whole_manifest_frozen.json").json(); rows=[r for r in manifest["plants"] if r["qa_status"] in {"accept","partial"}]
    device="cuda" if torch.cuda.is_available() else "cpu"
    proc=Sam2Processor.from_pretrained(SAM_ID,token=os.environ.get("HF_TOKEN")); sam=Sam2Model.from_pretrained(SAM_ID,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16 if device=="cuda" else torch.float32).to(device).eval(); samrev=getattr(sam.config,"_commit_hash",None) or "unknown"
    enriched=[]; masked_cache={}
    for i,r in enumerate(rows,1):
        raw=get(BRIDGE+r["crop_path"]).content; im=Image.open(io.BytesIO(raw)).convert("RGB"); cb=r["crop_bbox"]; b=r["bbox"]
        rel=[max(0,b["x"]-cb["x"]),max(0,b["y"]-cb["y"]),min(im.width,b["x"]+b["w"]-cb["x"]),min(im.height,b["y"]+b["h"]-cb["y"])]
        inputs=proc(images=im,input_boxes=[[rel]],return_tensors="pt").to(device)
        with torch.inference_mode(): out=sam(**inputs,multimask_output=True)
        mask,choice,score=select_mask(out,proc,inputs); mask=clean_mask(mask,rel); frac=float(mask.mean()); border=float(np.concatenate([mask[0],mask[-1],mask[:,0],mask[:,-1]]).mean()); valid=.01<=frac<=.90 and not (border>.85 and frac>.70)
        arr=np.array(im); white=np.full_like(arr,255); white[mask]=arr[mask]; mim=Image.fromarray(white); mbytes=png(mim); mpath=f"{PREFIX}/masked/{r['plant_id']}.png"; upload(mpath,"image/png",mbytes); masked_cache[r["plant_id"]]=mim
        y=int(round(im.height*float(r.get("root_boundary_y_1000") or 1000)/1000)); y=max(1,min(im.height,y)); yc=min(im.height,y+round(im.height*.05)); above=mim.crop((0,0,im.width,y)); context=mim.crop((0,0,im.width,yc)); abytes=png(above); cbytes=png(context); ap=f"{PREFIX}/above_strict/{r['plant_id']}.png"; cp=f"{PREFIX}/above_context/{r['plant_id']}.png"; upload(ap,"image/png",abytes); upload(cp,"image/png",cbytes)
        q={**r,"mask_model":SAM_ID,"mask_revision":samrev,"mask_choice":choice,"mask_iou_score":score,"mask_area_fraction":frac,"mask_border_fraction":border,"mask_valid":valid,"masked_crop_path":mpath,"masked_crop_sha256":sha(mbytes),"above_strict_path":ap,"above_strict_sha256":sha(abytes),"above_context_path":cp,"above_context_sha256":sha(cbytes),"reproductive_structures_raw_hint":r.get("reproductive_structures",[]),"reproductive_structures":[]}; enriched.append(q)
        print(json.dumps({"event":"mask","n":i,"total":len(rows),"plant":r["plant_id"],"valid":valid,"area":round(frac,4)},sort_keys=True),flush=True)
    del sam,proc; gc.collect(); torch.cuda.empty_cache()
    P=H["AutoProcessor"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),min_pixels=256*28*28,max_pixels=1280*28*28)
    M=H["Qwen2_5_VLForConditionalGeneration"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval(); qrev=getattr(M.config,"_commit_hash",None) or "unknown"
    for i,r in enumerate(enriched,1):
        if not r["mask_valid"]: continue
        im=masked_cache[r["plant_id"]]; ans,rawans=infer(M,P,[im],MASKED_REPRO,650); structs=ans.get("reproductive_structures",[]) if isinstance(ans,dict) and isinstance(ans.get("reproductive_structures"),list) else []
        for z in structs[:5]:
            if not isinstance(z,dict) or str(z.get("class","")).lower() not in {"flower","flower_head","inflorescence","bud","fruit","seed_head"}: continue
            rb=box(z.get("bbox_1000") or z.get("bbox"),im.width,im.height,14)
            if not rb: continue
            rcb=expand(rb,im.width,im.height,.12,.12,.12,.12); rim=im.crop(rcb); qa,qaraw=infer(M,P,[im,rim],ns["RQA"],450); st,cls,conf,reason=rqa(qa); rid=f"{r['plant_id']}_mr{len(r['reproductive_structures']):02d}"; data=png(rim); path=f"{PREFIX}/masked_reproductive/{rid}.png"; upload(path,"image/png",data)
            r["reproductive_structures"].append({"repro_id":rid,"proposal_index":z.get("proposal_index"),"proposed_class":z.get("class"),"proposal_confidence":ns["num"](z.get("confidence")),"proposal_description":z.get("description"),"bbox":bd(rb),"crop_bbox":bd(rcb),"crop_path":path,"crop_sha256":sha(data),"qa_status":st,"qa_class":cls,"qa_confidence":conf,"qa_reason":reason,"qa_raw":qa,"qa_response_raw":qaraw})
        print(json.dumps({"event":"masked_repro","n":i,"total":len(enriched),"plant":r["plant_id"],"proposals":len(r["reproductive_structures"])},sort_keys=True),flush=True)
    out={"protocol_id":manifest["protocol_id"],"whole_manifest_sha256":manifest["whole_manifest_sha256"],"sam_model":SAM_ID,"sam_revision":samrev,"qwen_model":MODEL,"qwen_revision":qrev,"mask_policy":{"prompt":"frozen whole-plant relative bbox","component_min_fraction":.00015,"valid_area":[.01,.90],"border_failure":{"border":.85,"area":.70}},"plants":enriched}
    out["counts"]={"broad_plants":len(enriched),"valid_masks":sum(x["mask_valid"] for x in enriched),"invalid_masks":sum(not x["mask_valid"] for x in enriched),"strict_whole":sum(x["qa_status"]=="accept" and x["mask_valid"] for x in enriched),"masked_reproductive":sum(len(x["reproductive_structures"]) for x in enriched),"repro_accept":sum(z["qa_status"]=="accept" for x in enriched for z in x["reproductive_structures"]),"repro_partial":sum(z["qa_status"]=="partial" for x in enriched for z in x["reproductive_structures"])}
    out["channel_manifest_sha256"]=csha(out); out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); upload(PREFIX+"/channels_manifest_frozen.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode()); upload(PREFIX+"/channels_report.json","application/json",json.dumps({k:out[k] for k in ("protocol_id","whole_manifest_sha256","channel_manifest_sha256","sam_model","sam_revision","qwen_model","qwen_revision","counts")},indent=2,sort_keys=True).encode()); print("RESULT_JSON="+json.dumps({"channel_manifest_sha256":out["channel_manifest_sha256"],"counts":out["counts"]},sort_keys=True),flush=True)
    del M,P; gc.collect(); torch.cuda.empty_cache()
if __name__=="__main__": main()
