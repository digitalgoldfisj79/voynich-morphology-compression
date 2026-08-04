#!/usr/bin/env python3
"""Paginated checkpoint-preserving extraction finalizer (v0.2)."""
from __future__ import annotations
import base64,csv,hashlib,io,json,math,time,zipfile
from pathlib import Path
from typing import Any
import requests
from PIL import Image,ImageDraw,ImageOps
PROTOCOL="CLM5905-VMS-RF-0.1-20260804";SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co";BRIDGE=SUPA+"/storage/v1/object/public/bridge/";UPLOAD=SUPA+"/functions/v1/clm5905-upload-v01";PREFIX="clm5905_v01/extraction";S=requests.Session();S.headers["User-Agent"]="CLM5905Finalizer/0.2"
def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def csha(x:Any)->str:return sha(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode())
def get(u:str)->bytes:
 r=S.get(u,timeout=300);r.raise_for_status();return r.content
def upload(path:str,typ:str,data:bytes)->None:
 r=S.post(UPLOAD,json={"path":path,"content_type":typ,"data_b64":base64.b64encode(data).decode()},timeout=360);r.raise_for_status();j=r.json()
 if j.get("error"):raise RuntimeError(j)
def png(im:Image.Image)->bytes:
 b=io.BytesIO();im.convert("RGB").save(b,"PNG",optimize=True);return b.getvalue()
def pages(records:list[dict[str,Any]],key:str,title:str,page_size:int=120)->list[tuple[str,bytes]]:
 available=[r for r in records if r.get(key)];out=[];cols=8;cw,ch=160,182
 for pi in range(math.ceil(len(available)/page_size)):
  chunk=available[pi*page_size:(pi+1)*page_size];nr=max(1,math.ceil(len(chunk)/cols));sheet=Image.new("RGB",(cols*cw,40+nr*ch),"white");d=ImageDraw.Draw(sheet);d.text((8,11),f"{title} — page {pi+1}/{math.ceil(len(available)/page_size)}",fill="black")
  for i,r in enumerate(chunk):
   try:im=Image.open(io.BytesIO(get(BRIDGE+r[key]))).convert("RGB")
   except Exception:continue
   x,y=(i%cols)*cw,40+(i//cols)*ch;t=ImageOps.contain(im,(cw-10,ch-34));sheet.paste(t,(x+(cw-t.width)//2,y));d.text((x+4,y+ch-30),f"#{r.get('illustration_number')} {r.get('folio','')}",fill="black");d.text((x+4,y+ch-15),str(r.get('root_status') or r.get('status') or '')[:18],fill="black")
  out.append((f"page_{pi+1:02d}.png",png(sheet)))
 return out
def main()->None:
 report=json.loads(get(BRIDGE+f"{PREFIX}/checkpoint.json"));records=report.get("records",[])
 if report.get("protocol_id")!=PROTOCOL or report.get("progress",{}).get("complete")!=198 or len(records)!=198:raise RuntimeError("complete checkpoint unavailable")
 report["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());report["counts"]={"illustrations":198,"whole_accept":sum(r.get("whole_status")=="accept" for r in records),"whole_partial":sum(r.get("whole_status")=="partial" for r in records),"root_accept":sum(r.get("root_status")=="accept" and bool(r.get("root_crop_path")) for r in records),"root_partial":sum(r.get("root_status")=="partial" and bool(r.get("root_crop_path")) for r in records),"flower_accept":sum(z.get("status")=="accept" and z.get("class") in {"flower","flower_head","inflorescence"} and bool(z.get("crop_path")) for r in records for z in r.get("reproductive",[])),"flower_broad":sum(z.get("status") in {"accept","partial"} and z.get("class") in {"flower","flower_head","inflorescence","bud"} and bool(z.get("crop_path")) for r in records for z in r.get("reproductive",[])),"errors":len(report.get("errors",[]))};report["localiser_cost_usd"]=float(sum(float(r.get("localiser_cost_usd") or 0) for r in records));report.pop("extraction_manifest_sha256",None);report["extraction_manifest_sha256"]=csha(report)
 manifest=json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False).encode();upload(f"{PREFIX}/extraction_manifest_frozen.json","application/json",manifest)
 out=Path("/tmp/clm5905_finalize_v02");out.mkdir(exist_ok=True);csvp=out/"plants.csv";fields=["illustration_number","folio","label","whole_status","whole_confidence","root_status","root_confidence","root_description","root_crop_path","root_masked_path","reproductive_count","source_image_url","source_image_sha256"]
 with csvp.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in records:w.writerow({"illustration_number":r.get("illustration_number"),"folio":r.get("folio"),"label":r.get("label"),"whole_status":r.get("whole_status"),"whole_confidence":r.get("whole_confidence"),"root_status":r.get("root_status"),"root_confidence":r.get("root_confidence"),"root_description":r.get("root_description"),"root_crop_path":r.get("root_crop_path"),"root_masked_path":r.get("root_masked_path"),"reproductive_count":len(r.get("reproductive",[])),"source_image_url":r.get("image_url"),"source_image_sha256":r.get("source_image_sha256")})
 repro=[]
 for r in records:
  for z in r.get("reproductive",[]):repro.append({**r,**z,"root_status":z.get("status")})
 qa=[]
 for family,recs,key,title in [("roots",records,"root_crop_path","BSB Clm 5905 — frozen root crops"),("reproductive",repro,"crop_path","BSB Clm 5905 — frozen reproductive crops")]:
  for name,data in pages(recs,key,title):
   p=f"{PREFIX}/qa/{family}/{name}";upload(p,"image/png",data);qa.append({"family":family,"path":p,"bytes":len(data),"sha256":sha(data)})
 qa_index={"protocol_id":PROTOCOL,"pages":qa};qa_index["sha256"]=csha(qa_index);qa_data=json.dumps(qa_index,indent=2,sort_keys=True).encode();upload(f"{PREFIX}/QA_INDEX.json","application/json",qa_data)
 md=f"""# BSB Clm 5905 extraction report

Protocol: `{PROTOCOL}`

- Frozen illustrations: 198
- Whole accept / partial: {report['counts']['whole_accept']} / {report['counts']['whole_partial']}
- Root strict / partial: {report['counts']['root_accept']} / {report['counts']['root_partial']}
- Strict flowers: {report['counts']['flower_accept']}
- Broad flower structures: {report['counts']['flower_broad']}
- Extraction errors: {report['counts']['errors']}
- Recorded localiser cost: USD {report['localiser_cost_usd']:.4f}
- QA pages: {len(qa)}
- Extraction manifest SHA-256: `{report['extraction_manifest_sha256']}`

No Voynich image or similarity was opened in this phase.
""";upload(f"{PREFIX}/EXTRACTION_REPORT.md","text/markdown",md.encode())
 source=get(BRIDGE+f"{PREFIX}/source_freeze.json");ident=get(BRIDGE+f"{PREFIX}/fischer_identifications.json");b=io.BytesIO()
 with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  z.writestr("00_PROTOCOL/PROTOCOL_ID.txt",PROTOCOL+"\n");z.writestr("01_SOURCE/source_freeze.json",source);z.writestr("01_SOURCE/fischer_identifications.json",ident);z.writestr("02_EXTRACTION/extraction_manifest_frozen.json",manifest);z.writestr("02_EXTRACTION/plants.csv",csvp.read_bytes());z.writestr("03_QA/QA_INDEX.json",qa_data);z.writestr("EXTRACTION_REPORT.md",md)
 data=b.getvalue();upload(f"{PREFIX}/CLM5905_EXTRACTION_CORE.zip","application/zip",data);print("RESULT_JSON="+json.dumps({"protocol_id":PROTOCOL,"counts":report["counts"],"manifest_sha256":report["extraction_manifest_sha256"],"qa_pages":len(qa),"bundle_sha256":sha(data),"bundle_bytes":len(data),"bundle_url":BRIDGE+f"{PREFIX}/CLM5905_EXTRACTION_CORE.zip"},sort_keys=True),flush=True)
if __name__=="__main__":main()
