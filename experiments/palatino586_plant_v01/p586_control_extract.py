#!/usr/bin/env python3
"""Fixed-control page screening and whole-plant extraction; no similarities."""
from __future__ import annotations
import io,json,os,time,requests,torch
from PIL import Image,ImageOps

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/1f57a841e79f5677726025a2ecb70fe5901e6cd5/experiments/palatino586_plant_v01/p586_target_extract.py"
code=requests.get(BASE,timeout=120).text; ns={"__name__":"p586_target_lib"}; exec(compile(code,BASE,"exec"),ns)
get,upload,png,jpg,sha,csha,box,iou,bd,expand,infer,plants,qa,channel,rqa= [ns[k] for k in ("get","upload","png","jpg","sha","csha","box","iou","bd","expand","infer","plants","qa","channel","rqa")]
H=ns["H"]; MODEL=ns["MODEL"]; DETECT=ns["DETECT"]; QA=ns["QA"]; CHANNEL=ns["CHANNEL"]; RQA=ns["RQA"]
CID=os.environ["CONTROL_ID"]
CORPUS=f"https://ymaqlcfjmdwncdbjprmw.supabase.co/functions/v1/p586-plant-corpus-v01?kind=all_pages&id={CID}"
PREFIX=f"p586_plant_v01/controls/{CID}"
SCREEN='''Return strict JSON only: {"contains_botanical_plant":true,"candidate_count":1,"confidence":0.0,"reason":"brief"}. True only when this manuscript page visibly contains at least one coherent botanical plant illustration with stems/leaves/roots/flowers, not merely floral decoration, marginal ornament, isolated leaf motifs, text, people or animals.'''

def resize(im,maxside):
    if max(im.size)<=maxside:return im
    scale=maxside/max(im.size); return im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)

def screen_answer(a):
    if not isinstance(a,dict): return False,0,0.0,"malformed"
    return bool(a.get("contains_botanical_plant")),int(ns["num"](a.get("candidate_count"),0)),ns["num"](a.get("confidence")),str(a.get("reason", ""))

def save(out):
    out["counts"]={"pages_screened":len(out["pages"]),"screen_positive":sum(p.get("screen_admit") for p in out["pages"]),"whole_proposals":len(out["plants"]),**{k:sum(x["qa_status"]==k for x in out["plants"]) for k in ("accept","partial","reject","uncertain")}}
    out["counts"]["broad"]=out["counts"]["accept"]+out["counts"]["partial"]
    upload(PREFIX+"/checkpoint.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode())

def main():
    source=get(CORPUS).json(); rows=source["rows"]
    P=H["AutoProcessor"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),min_pixels=256*28*28,max_pixels=1280*28*28)
    M=H["Qwen2_5_VLForConditionalGeneration"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval(); rev=getattr(M.config,"_commit_hash",None) or "unknown"
    out={"protocol_id":"P586-VMS-PLANT-0.1-20260803","amendment":"PREREGISTRATION_AMENDMENT_01","manuscript_id":CID,"page_registry_count":len(rows),"detector_model":MODEL,"detector_revision":rev,"screen_threshold":.70,"broad_cap":20,"pages":[],"plants":[]}
    for pi,row in enumerate(rows,1):
        if out.get("counts",{}).get("broad",0)>=20: break
        rec={"seq":row["seq"],"slug":row["slug"],"folio_canonical":row.get("folio_canonical"),"source_image_url":row["image_url"]}
        try:
            raw=get(row["image_url"]).content; page=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB"); rec.update(source_image_sha256=sha(raw),image_width=page.width,image_height=page.height)
            sim=resize(page,768); a,araw=infer(M,P,[sim],SCREEN,220); yes,n,c,why=screen_answer(a); admit=yes and c>=.70; rec.update(screen_contains_botanical_plant=yes,screen_candidate_count=n,screen_confidence=c,screen_reason=why,screen_admit=admit,screen_raw=a,screen_response_raw=araw)
            if admit:
                detim=resize(page,1600); sx=page.width/detim.width; sy=page.height/detim.height; da,draw=infer(M,P,[detim],DETECT,900); cand=plants(da); parsed=[]
                for pos,z in enumerate(cand):
                    b=box(z.get("bbox_1000") or z.get("bbox"),detim.width,detim.height,45) if isinstance(z,dict) else None
                    if b:
                        b=(round(b[0]*sx),round(b[1]*sy),round(b[2]*sx),round(b[3]*sy)); parsed.append((pos,z,b,ns["num"](z.get("confidence"))))
                parsed.sort(key=lambda q:(-q[3],q[0])); keep=[]
                for q in parsed:
                    if not any(iou(q[2],x[2])>.65 for x in keep):keep.append(q)
                keep.sort(key=lambda q:(q[2][1],q[2][0],q[0])); rec.update(raw_candidates=len(cand),valid_candidates=len(parsed),unique_candidates=len(keep),detector_raw=draw)
                for j,(pos,z,b,dc) in enumerate(keep):
                    cb=expand(b,page.width,page.height); crop=page.crop(cb); pid=f"s{int(row['seq']):04d}_p{j:02d}"; qans,qraw=infer(M,P,[ns["marked"](page,cb,pid),crop],QA,550); st,qc,reason=qa(qans); data=png(crop); path=f"{PREFIX}/whole/{pid}.png"; upload(path,"image/png",data)
                    plant={"plant_id":pid,"manuscript_id":CID,"seq":row["seq"],"slug":row["slug"],"folio_canonical":row.get("folio_canonical"),"source_image_url":row["image_url"],"plant_index":j,"detector_position":pos,"detector_confidence":dc,"detector_complete":z.get("complete"),"detector_has_visible_root":z.get("has_visible_root"),"detector_description":z.get("description"),"bbox":bd(b),"crop_bbox":bd(cb),"crop_path":path,"crop_sha256":sha(data),"qa_status":st,"qa_confidence":qc,"qa_reason":reason,"qa_raw":qans,"qa_response_raw":qraw,"reproductive_structures":[]}
                    if st in {"accept","partial"}:
                        ca,craw=infer(M,P,[crop],CHANNEL,750); roots,y,yc,yr,structs=channel(ca); plant.update(has_visible_roots=roots,root_boundary_y_1000=y,root_boundary_confidence=yc,root_boundary_reason=yr,channel_raw=ca,channel_response_raw=craw)
                    else: plant.update(has_visible_roots=None,root_boundary_y_1000=None,root_boundary_confidence=None,root_boundary_reason="excluded by whole QA")
                    out["plants"].append(plant)
                    save(out)
                    if out["counts"]["broad"]>=20: break
            out["pages"].append(rec)
        except Exception as e: rec.update(screen_admit=False,error=f"{type(e).__name__}: {e}"); out["pages"].append(rec)
        save(out); print(json.dumps({"event":"control_page","manuscript":CID,"page":pi,"registry":len(rows),"seq":row["seq"],"counts":out["counts"]},sort_keys=True),flush=True)
    out["plants"].sort(key=lambda x:(int(x["seq"]),int(x["plant_index"]))); out["whole_manifest_sha256"]=csha({k:out[k] for k in ("protocol_id","amendment","manuscript_id","page_registry_count","detector_model","detector_revision","screen_threshold","broad_cap","pages","plants")}); out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); save(out); upload(PREFIX+"/whole_manifest_frozen.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode()); print("RESULT_JSON="+json.dumps({"manuscript_id":CID,"whole_manifest_sha256":out["whole_manifest_sha256"],"counts":out["counts"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
