#!/usr/bin/env python3
"""Non-inferential SAM2 contract diagnostic on the first eight Voynich broad crops."""
from __future__ import annotations
import io,json,os,requests,torch
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from transformers import Sam2Model,Sam2Processor
BASE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/1f57a841e79f5677726025a2ecb70fe5901e6cd5/experiments/palatino586_plant_v01/p586_target_extract.py"
code=requests.get(BASE,timeout=120).text;ns={"__name__":"lib"};exec(compile(code,BASE,"exec"),ns)
get,upload,png=ns["get"],ns["upload"],ns["png"];BRIDGE=ns["BRIDGE"]
MID="voynich";PREFIX="p586_plant_v01/controls/voynich";SAM="facebook/sam2.1-hiera-large"

def relbox(p,im):
 cb=p.get("crop_bbox");b=p.get("bbox")
 if not cb or not b:return [0,0,im.width,im.height]
 return [max(0,b["x"]-cb["x"]),max(0,b["y"]-cb["y"]),min(im.width,b["x"]+b["w"]-cb["x"]),min(im.height,b["y"]+b["h"]-cb["y"])]
def overlay(im,mask,label):
 a=np.asarray(im).copy();m=np.asarray(mask,dtype=bool);a[m]=(0.55*a[m]+0.45*np.array([255,0,0])).astype(np.uint8);o=Image.fromarray(a);d=ImageDraw.Draw(o);d.rectangle((2,2,min(o.width-2,500),28),fill="white");d.text((5,7),label,fill="black");return o

def main():
 man=get(BRIDGE+PREFIX+"/whole_manifest_frozen.json").json();rows=[p for p in man["plants"] if p.get("qa_status") in {"accept","partial"}][:8]
 dev="cuda" if torch.cuda.is_available() else "cpu";proc=Sam2Processor.from_pretrained(SAM,token=os.environ.get("HF_TOKEN"));model=Sam2Model.from_pretrained(SAM,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.float32).to(dev).eval();report=[];panels=[]
 for p in rows:
  im=Image.open(io.BytesIO(get(BRIDGE+p["crop_path"]).content)).convert("RGB");rb=relbox(p,im);inputs=proc(images=im,input_boxes=[[rb]],return_tensors="pt").to(dev)
  with torch.inference_mode():out=model(**inputs,multimask_output=True)
  masks=proc.post_process_masks(out.pred_masks.cpu(),inputs["original_sizes"].cpu())[0]
  while masks.ndim>3:masks=masks[0]
  scores=out.iou_scores.detach().cpu().reshape(-1);obj=out.object_score_logits.detach().cpu().reshape(-1)
  rec={"plant_id":p["plant_id"],"image_size":im.size,"relative_box":rb,"raw_pred_shape":list(out.pred_masks.shape),"post_shape":list(masks.shape),"iou_scores":[float(x) for x in scores],"object_scores":[float(x) for x in obj],"mask_area_fractions":[float((m.numpy()>0).mean()) for m in masks]};report.append(rec)
  variants=[]
  for j,m in enumerate(masks):variants.append(overlay(im,m.numpy()>0,f"{p['plant_id']} m{j} area={rec['mask_area_fractions'][j]:.3f} iou={rec['iou_scores'][j]:.3f}"))
  panels.append((im,variants))
 print(json.dumps(report,indent=2),flush=True)
 # Contact sheet: source then three masks per row
 cw,ch=280,330;sheet=Image.new("RGB",(4*cw,len(panels)*ch),"white")
 for i,(im,vars) in enumerate(panels):
  for j,q in enumerate([im]+vars):
   z=ImageOps.contain(q,(cw-8,ch-8));sheet.paste(z,(j*cw+(cw-z.width)//2,i*ch+(ch-z.height)//2))
 upload("p586_plant_v01/diagnostics/sam2_voynich_first8.png","image/png",png(sheet));upload("p586_plant_v01/diagnostics/sam2_voynich_first8.json","application/json",json.dumps(report,indent=2).encode());print("DIAGNOSTIC_PATH=p586_plant_v01/diagnostics/sam2_voynich_first8.png",flush=True)
if __name__=="__main__":main()
