#!/usr/bin/env python3
"""Apply one frozen SAM2/masked-reproductive policy to any frozen corpus."""
from __future__ import annotations
import gc,io,json,os,time,requests,cv2,numpy as np,torch
from PIL import Image
from transformers import Sam2Model,Sam2Processor
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/1f57a841e79f5677726025a2ecb70fe5901e6cd5/experiments/palatino586_plant_v01/p586_target_extract.py"
code=requests.get(BASE,timeout=120).text; ns={"__name__":"targetlib"}; exec(compile(code,BASE,"exec"),ns)
get,upload,png,sha,csha,box,expand,bd,infer,rqa=[ns[k] for k in ("get","upload","png","sha","csha","box","expand","bd","infer","rqa")]
H=ns["H"]; MODEL=ns["MODEL"]; BRIDGE=ns["BRIDGE"]; CID=os.environ["CORPUS_ID"]
PREFIX="p586_plant_v01/target" if CID=="bncf_palatino_586" else f"p586_plant_v01/controls/{CID}"
SAM_ID="facebook/sam2.1-hiera-large"
PROMPT='''This is a white-background masked medieval plant. Return strict JSON only: {"reproductive_structures":[{"proposal_index":0,"class":"flower|flower_head|inflorescence|bud|fruit|seed_head","bbox_1000":[x0,y0,x1,y1],"confidence":0.0,"description":"brief"}]}. Propose at most five coherent reproductive structures. Exclude ordinary leaves, stems, roots, text and background.'''

def clean(mask,rel):
 m=(mask>0).astype(np.uint8);h,w=m.shape;n,lab,stats,_=cv2.connectedComponentsWithStats(m,8);keep=np.zeros_like(m);x0,y0,x1,y1=rel
 for i in range(1,n):
  x,y,cw,ch,a=stats[i];inter=max(0,min(x+cw,x1)-max(x,x0))*max(0,min(y+ch,y1)-max(y,y0))
  if a>=max(12,int(h*w*.00015)) and (inter>0 or a>=h*w*.002):keep[lab==i]=1
 k=max(3,(min(h,w)//120)|1);return cv2.morphologyEx(keep,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8)).astype(bool)

def choose(out,proc,inputs):
 ms=proc.post_process_masks(out.pred_masks.detach().cpu(),inputs["original_sizes"].detach().cpu())[0];sc=out.iou_scores.detach().cpu().reshape(-1)
 if ms.ndim==4:ms=ms[0]
 if ms.ndim==2:return ms.numpy()>0,0,float(sc[0])
 j=int(torch.argmax(sc[:ms.shape[0]]));return ms[j].numpy()>0,j,float(sc[j])

def main():
 man=get(BRIDGE+PREFIX+"/whole_manifest_frozen.json").json();rows=[r for r in man["plants"] if r.get("qa_status") in {"accept","partial"}]
 dev="cuda" if torch.cuda.is_available() else "cpu";proc=Sam2Processor.from_pretrained(SAM_ID,token=os.environ.get("HF_TOKEN"));sam=Sam2Model.from_pretrained(SAM_ID,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16 if dev=="cuda" else torch.float32).to(dev).eval();srev=getattr(sam.config,"_commit_hash",None) or "unknown";en=[];cache={}
 for i,r in enumerate(rows,1):
  try:
   raw=get(BRIDGE+r["crop_path"]).content;im=Image.open(io.BytesIO(raw)).convert("RGB");cb=r.get("crop_bbox");b=r.get("bbox")
   if cb and b:rel=[max(0,b["x"]-cb["x"]),max(0,b["y"]-cb["y"]),min(im.width,b["x"]+b["w"]-cb["x"]),min(im.height,b["y"]+b["h"]-cb["y"])]
   else:rel=[0,0,im.width,im.height]
   inputs=proc(images=im,input_boxes=[[rel]],return_tensors="pt").to(dev)
   with torch.inference_mode():out=sam(**inputs,multimask_output=True)
   mask,ch,score=choose(out,proc,inputs);mask=clean(mask,rel);frac=float(mask.mean());border=float(np.concatenate([mask[0],mask[-1],mask[:,0],mask[:,-1]]).mean());valid=.01<=frac<=.90 and not(border>.85 and frac>.70)
   arr=np.array(im);white=np.full_like(arr,255);white[mask]=arr[mask];mim=Image.fromarray(white);mb=png(mim);mp=f"{PREFIX}/masked/{r['plant_id']}.png";upload(mp,"image/png",mb);cache[r["plant_id"]]=mim
   y=int(round(im.height*float(r.get("root_boundary_y_1000") or 1000)/1000));y=max(1,min(im.height,y));yc=min(im.height,y+round(im.height*.05));ai=mim.crop((0,0,im.width,y));ci=mim.crop((0,0,im.width,yc));ab=png(ai);ccb=png(ci);ap=f"{PREFIX}/above_strict/{r['plant_id']}.png";cp=f"{PREFIX}/above_context/{r['plant_id']}.png";upload(ap,"image/png",ab);upload(cp,"image/png",ccb)
   en.append({**r,"mask_model":SAM_ID,"mask_revision":srev,"mask_choice":ch,"mask_iou_score":score,"mask_area_fraction":frac,"mask_border_fraction":border,"mask_valid":valid,"masked_crop_path":mp,"masked_crop_sha256":sha(mb),"above_strict_path":ap,"above_strict_sha256":sha(ab),"above_context_path":cp,"above_context_sha256":sha(ccb),"reproductive_structures":[]})
  except Exception as e:en.append({**r,"mask_model":SAM_ID,"mask_revision":srev,"mask_valid":False,"mask_error":f"{type(e).__name__}: {e}","reproductive_structures":[]})
  print(json.dumps({"event":"mask","corpus":CID,"n":i,"total":len(rows),"valid":en[-1].get("mask_valid")},sort_keys=True),flush=True)
 del sam,proc;gc.collect();torch.cuda.empty_cache()
 P=H["AutoProcessor"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),min_pixels=256*28*28,max_pixels=1280*28*28);M=H["Qwen2_5_VLForConditionalGeneration"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval();qrev=getattr(M.config,"_commit_hash",None) or "unknown"
 for i,r in enumerate(en,1):
  if not r.get("mask_valid"):continue
  im=cache[r["plant_id"]];ans,rawans=infer(M,P,[im],PROMPT,650);structs=ans.get("reproductive_structures",[]) if isinstance(ans,dict) and isinstance(ans.get("reproductive_structures"),list) else []
  for z in structs[:5]:
   if not isinstance(z,dict) or str(z.get("class","")).lower() not in {"flower","flower_head","inflorescence","bud","fruit","seed_head"}:continue
   rb=box(z.get("bbox_1000") or z.get("bbox"),im.width,im.height,14)
   if not rb:continue
   rc=expand(rb,im.width,im.height,.12,.12,.12,.12);ri=im.crop(rc);qa,qaraw=infer(M,P,[im,ri],ns["RQA"],450);st,cl,cf,reason=rqa(qa);rid=f"{r['plant_id']}_mr{len(r['reproductive_structures']):02d}";data=png(ri);path=f"{PREFIX}/masked_reproductive/{rid}.png";upload(path,"image/png",data);r["reproductive_structures"].append({"repro_id":rid,"proposed_class":z.get("class"),"proposal_confidence":ns["num"](z.get("confidence")),"bbox":bd(rb),"crop_bbox":bd(rc),"crop_path":path,"crop_sha256":sha(data),"qa_status":st,"qa_class":cl,"qa_confidence":cf,"qa_reason":reason,"qa_raw":qa,"qa_response_raw":qaraw})
  print(json.dumps({"event":"masked_repro","corpus":CID,"n":i,"total":len(en),"repro":len(r["reproductive_structures"])},sort_keys=True),flush=True)
 out={"protocol_id":"P586-VMS-PLANT-0.1-20260803","corpus_id":CID,"whole_manifest_sha256":man["whole_manifest_sha256"],"sam_model":SAM_ID,"sam_revision":srev,"qwen_model":MODEL,"qwen_revision":qrev,"plants":en};out["counts"]={"broad_plants":len(en),"valid_masks":sum(x.get("mask_valid",False) for x in en),"invalid_masks":sum(not x.get("mask_valid",False) for x in en),"strict_whole":sum(x.get("qa_status")=="accept" and x.get("mask_valid",False) for x in en),"reproductive":sum(len(x["reproductive_structures"]) for x in en),"repro_accept":sum(z["qa_status"]=="accept" for x in en for z in x["reproductive_structures"]),"repro_partial":sum(z["qa_status"]=="partial" for x in en for z in x["reproductive_structures"])};out["channel_manifest_sha256"]=csha(out);out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());upload(PREFIX+"/channels_manifest_frozen.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode());print("RESULT_JSON="+json.dumps({"corpus_id":CID,"channel_manifest_sha256":out["channel_manifest_sha256"],"counts":out["counts"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
