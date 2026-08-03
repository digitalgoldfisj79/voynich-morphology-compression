#!/usr/bin/env python3
"""Frozen colour-mask visual audit and reproductive-structure extraction."""
from __future__ import annotations
import hashlib, io, json, math, os, time, requests, torch
from PIL import Image,ImageDraw,ImageOps

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/1f57a841e79f5677726025a2ecb70fe5901e6cd5/experiments/palatino586_plant_v01/p586_target_extract.py"
code=requests.get(BASE,timeout=120).text;ns={"__name__":"p586lib"};exec(compile(code,BASE,"exec"),ns)
get,upload,png,jpg,sha,csha,box,expand,bd,infer,rqa=[ns[k] for k in ("get","upload","png","jpg","sha","csha","box","expand","bd","infer","rqa")]
H=ns["H"];MODEL=ns["MODEL"];BRIDGE=ns["BRIDGE"]
PROTOCOL="P586-VMS-PLANT-0.1-20260803";TARGET="bncf_palatino_586";CID=os.environ["CORPUS_ID"]
PREFIX="p586_plant_v01/target" if CID==TARGET else f"p586_plant_v01/controls/{CID}"
MASK_AUDIT='''Images show the same frozen medieval whole-plant crop: Image 1 ordinary source; Image 2 quantised colour mask (green above-ground plant, red root/bulb/rhizome, white background); Image 3 source pixels retained under that mask. Return strict JSON only: {"status":"pass|partial|fail|uncertain","confidence":0.0,"foreground_precision":"good|mixed|poor","foreground_recall":"good|mixed|poor","root_split":"good|mixed|poor|not_applicable","reason":"brief"}. Pass only when the mask captures the coherent plant sufficiently for morphology comparison without substantial parchment, text, or neighbouring objects. Partial is useful but materially incomplete/contaminated. Fail is not a usable plant mask. Judge segmentation, not artistic quality.'''
MASKED_REPRO='''This is a white-background masked medieval plant. Return strict JSON only: {"reproductive_structures":[{"proposal_index":0,"class":"flower|flower_head|inflorescence|bud|fruit|seed_head","bbox_1000":[x0,y0,x1,y1],"confidence":0.0,"description":"brief"}]}. Propose at most five coherent reproductive structures. Exclude ordinary leaves, stems, roots, text and background.'''

def audit_selected(pid:str)->bool:
    h=hashlib.sha256(f"{PROTOCOL}|{CID}|{pid}".encode()).digest();return int.from_bytes(h[:4],"big") < int(.10*(2**32))
def parse_audit(a):
    if not isinstance(a,dict):return "uncertain",0.0,"malformed",{}
    st=str(a.get("status","uncertain")).lower();st=st if st in {"pass","partial","fail","uncertain"} else "uncertain";cf=ns["num"](a.get("confidence"));return st,cf,str(a.get("reason","")),a

def load_checkpoint(whole_sha,color_sha):
    try:
        x=get(BRIDGE+PREFIX+"/channel_qwen_checkpoint.json").json()
        if x.get("whole_manifest_sha256")!=whole_sha or x.get("color_mask_manifest_sha256")!=color_sha:raise RuntimeError("checkpoint mismatch")
        return x
    except Exception:return {"protocol_id":PROTOCOL,"corpus_id":CID,"whole_manifest_sha256":whole_sha,"color_mask_manifest_sha256":color_sha,"qwen_model":MODEL,"plants":[]}
def save(out):
    ps=out["plants"];aud=[p for p in ps if p.get("mask_audit_required") and p.get("mask_audit_status") not in {None,"not_auditable"}];av=[p for p in aud if p.get("area_valid")]
    out["counts"]={"plants":len(ps),"mask_success":sum(p.get("mask_status")=="success" for p in ps),"area_valid":sum(p.get("area_valid",False) for p in ps),"audited":len(aud),"audited_area_valid":len(av),"audited_valid_fail":sum(p.get("mask_audit_status")=="fail" for p in av),"mask_broad_valid":sum(p.get("mask_valid",False) for p in ps),"mask_strict_valid":sum(p.get("mask_strict_valid",False) for p in ps),"above_broad_valid":sum(p.get("above_valid_final",False) for p in ps),"reproductive_proposals":sum(len(p.get("reproductive_structures",[])) for p in ps),"repro_accept":sum(z.get("qa_status")=="accept" for p in ps for z in p.get("reproductive_structures",[])),"repro_partial":sum(z.get("qa_status")=="partial" for p in ps for z in p.get("reproductive_structures",[]))}
    out["audited_area_valid_fail_rate"]=out["counts"]["audited_valid_fail"]/max(1,out["counts"]["audited_area_valid"])
    out["method_reliable_under_frozen_gate"]=not(CID in {TARGET,"voynich"} and out["counts"]["audited_area_valid"]>0 and out["audited_area_valid_fail_rate"]>.20)
    upload(PREFIX+"/channel_qwen_checkpoint.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode())

def contact(rows,title):
    cw,ch=300,300;cols=3;out=Image.new("RGB",(cols*cw,45+len(rows)*ch),"white");d=ImageDraw.Draw(out);d.text((8,12),title,fill="black")
    for i,(p,ims) in enumerate(rows):
        y=45+i*ch
        for j,im in enumerate(ims):
            q=ImageOps.contain(im,(cw-12,ch-45));out.paste(q,(j*cw+(cw-q.width)//2,y+25+(ch-45-q.height)//2))
        d.text((5,y+4),f"{p['plant_id']} {p.get('mask_audit_status')} {p.get('mask_audit_confidence',0):.2f}",fill="black")
    return out

def main():
    whole=get(BRIDGE+PREFIX+"/whole_manifest_frozen.json").json();colors=get(BRIDGE+PREFIX+"/color_masks_frozen.json").json()
    if not colors.get("complete"):raise RuntimeError("colour-mask phase is not complete")
    cmap={x["plant_id"]:x for x in colors["records"]};broad=[x for x in whole["plants"] if x.get("qa_status") in {"accept","partial"}]
    out=load_checkpoint(whole["whole_manifest_sha256"],colors["color_mask_manifest_sha256"]);existing={x["plant_id"]:x for x in out["plants"]}
    P=H["AutoProcessor"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),min_pixels=256*28*28,max_pixels=1280*28*28)
    M=H["Qwen2_5_VLForConditionalGeneration"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval();qrev=getattr(M.config,"_commit_hash",None) or "unknown";out["qwen_revision"]=qrev
    audit_rows=[]
    for i,p in enumerate(broad,1):
        if existing.get(p["plant_id"],{}).get("qwen_complete"):continue
        m=cmap.get(p["plant_id"],{});rec={**p,"mask_status":m.get("status"),"mask_model":m.get("model"),"mask_prompt_sha256":m.get("prompt_sha256"),"raw_mask_path":m.get("raw_mask_path"),"raw_mask_sha256":m.get("raw_mask_sha256"),"quantized_mask_path":m.get("quantized_mask_path"),"quantized_mask_sha256":m.get("quantized_mask_sha256"),"mask_resized":m.get("mask_resized"),"green_fraction":m.get("green_fraction"),"red_fraction":m.get("red_fraction"),"foreground_fraction":m.get("foreground_fraction"),"mean_nearest_colour_distance":m.get("mean_nearest_colour_distance"),"p95_nearest_colour_distance":m.get("p95_nearest_colour_distance"),"area_valid":bool(m.get("area_valid")),"masked_crop_path":m.get("masked_crop_path"),"masked_crop_sha256":m.get("masked_crop_sha256"),"above_strict_path":m.get("above_strict_path"),"above_strict_sha256":m.get("above_strict_sha256"),"above_context_path":m.get("above_context_path"),"above_context_sha256":m.get("above_context_sha256"),"above_objective_valid":bool(m.get("above_valid")),"mask_audit_required":(not m.get("area_valid",False)) or audit_selected(p["plant_id"]),"reproductive_structures":[]}
        source=quant=masked=None
        if m.get("status")=="success":
            source=Image.open(io.BytesIO(get(BRIDGE+p["crop_path"]).content)).convert("RGB");quant=Image.open(io.BytesIO(get(BRIDGE+m["quantized_mask_path"]).content)).convert("RGB");masked=Image.open(io.BytesIO(get(BRIDGE+m["masked_crop_path"]).content)).convert("RGB")
        if rec["mask_audit_required"]:
            if source is None:rec.update(mask_audit_status="not_auditable",mask_audit_confidence=0.0,mask_audit_reason=m.get("error","implementation failure"))
            else:
                a,araw=infer(M,P,[source,quant,masked],MASK_AUDIT,500);st,cf,reason,parsed=parse_audit(a);rec.update(mask_audit_status=st,mask_audit_confidence=cf,mask_audit_reason=reason,mask_audit_raw=parsed,mask_audit_response_raw=araw);audit_rows.append((rec,[source,quant,masked]))
        else:rec.update(mask_audit_status="not_sampled",mask_audit_confidence=None,mask_audit_reason="objective-valid deterministic non-sample")
        audit_fail=rec["mask_audit_status"]=="fail";audit_nonpass=rec["mask_audit_status"] in {"partial","uncertain","fail"}
        rec["mask_valid"]=bool(m.get("area_valid")) and not audit_fail
        rec["mask_strict_valid"]=rec["mask_valid"] and p.get("qa_status")=="accept" and not audit_nonpass
        rec["above_valid_final"]=rec["mask_valid"] and bool(m.get("above_valid"))
        if rec["mask_valid"] and masked is not None:
            ans,rawans=infer(M,P,[masked],MASKED_REPRO,650);structs=ans.get("reproductive_structures",[]) if isinstance(ans,dict) and isinstance(ans.get("reproductive_structures"),list) else []
            for z in structs[:5]:
                if not isinstance(z,dict) or str(z.get("class","")).lower() not in {"flower","flower_head","inflorescence","bud","fruit","seed_head"}:continue
                rb=box(z.get("bbox_1000") or z.get("bbox"),masked.width,masked.height,14)
                if not rb:continue
                rcb=expand(rb,masked.width,masked.height,.12,.12,.12,.12);rim=masked.crop(rcb);qa,qaraw=infer(M,P,[masked,rim],ns["RQA"],450);st,cl,cf,reason=rqa(qa);rid=f"{p['plant_id']}_mr{len(rec['reproductive_structures']):02d}";data=png(rim);path=f"{PREFIX}/masked_reproductive/{rid}.png";upload(path,"image/png",data);rec["reproductive_structures"].append({"repro_id":rid,"proposed_class":z.get("class"),"proposal_confidence":ns["num"](z.get("confidence")),"proposal_description":z.get("description"),"bbox":bd(rb),"crop_bbox":bd(rcb),"crop_path":path,"crop_sha256":sha(data),"qa_status":st,"qa_class":cl,"qa_confidence":cf,"qa_reason":reason,"qa_raw":qa,"qa_response_raw":qaraw})
            rec["reproductive_response_raw"]=rawans
        rec["qwen_complete"]=True;existing[p["plant_id"]]=rec;out["plants"]=[existing[x["plant_id"]] for x in broad if x["plant_id"] in existing];save(out);print(json.dumps({"event":"qwen_channel","corpus":CID,"n":i,"total":len(broad),"plant":p["plant_id"],"audit":rec["mask_audit_status"],"mask_valid":rec["mask_valid"],"repro":len(rec["reproductive_structures"])},sort_keys=True),flush=True)
    out["plants"]=[existing[x["plant_id"]] for x in broad];save(out);out["channel_manifest_sha256"]=csha({k:out[k] for k in ("protocol_id","corpus_id","whole_manifest_sha256","color_mask_manifest_sha256","qwen_model","qwen_revision","plants")});out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());upload(PREFIX+"/channels_manifest_frozen.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode())
    if audit_rows:
        for start in range(0,len(audit_rows),30):upload(f"{PREFIX}/mask_audit_sheets/audit_{start//30:02d}.jpg","image/jpeg",jpg(contact(audit_rows[start:start+30],f"{CID} mask audit {start+1}-{min(len(audit_rows),start+30)}")))
    print("RESULT_JSON="+json.dumps({"corpus_id":CID,"channel_manifest_sha256":out["channel_manifest_sha256"],"counts":out["counts"],"fail_rate":out["audited_area_valid_fail_rate"],"method_reliable":out["method_reliable_under_frozen_gate"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
