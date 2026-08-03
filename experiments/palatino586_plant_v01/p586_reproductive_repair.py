#!/usr/bin/env python3
"""Repair the frozen reproductive-response interface and run batched visual QA.

No reproductive detector is rerun. Raw responses already frozen in the channel
manifest are parsed under Amendments 05-06, cropped, and adjudicated with the
same Qwen model used by the original channel runner.
"""
from __future__ import annotations
import gc, io, json, os, re, time, requests, torch
from PIL import Image

BASE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/1f57a841e79f5677726025a2ecb70fe5901e6cd5/experiments/palatino586_plant_v01/p586_target_extract.py"
code=requests.get(BASE,timeout=120).text; ns={"__name__":"p586lib"}; exec(compile(code,BASE,"exec"),ns)
get,upload,png,csha,sha,box,expand,bd,infer,rqa=[ns[k] for k in ("get","upload","png","csha","sha","box","expand","bd","infer","rqa")]
H=ns["H"]; MODEL=ns["MODEL"]; BRIDGE=ns["BRIDGE"]
PROTOCOL="P586-VMS-PLANT-0.1-20260803"; TARGET="bncf_palatino_586"; CID=os.environ["CORPUS_ID"]
PREFIX="p586_plant_v01/target" if CID==TARGET else f"p586_plant_v01/controls/{CID}"
ALLOWED={"flower","flower_head","inflorescence","bud","fruit","seed_head"}
PARSER_REVISION="repro-interface-v2-list-bbox2d"
QA_REVISION="repro-batch-qa-v1"
BATCH_QA='''Image 1 is the complete white-background masked medieval plant. Each later image is one proposed reproductive structure, in proposal order. Return strict JSON only: {"reviews":[{"proposal_index":0,"status":"accept|partial|reject|uncertain","class":"flower|flower_head|inflorescence|bud|fruit|seed_head|none","confidence":0.0,"attached_to_plant":true,"reason":"brief"}]}. Return exactly one review per proposal. Accept a coherent visible reproductive structure attached to the plant. Partial is genuine but materially cropped or ambiguous. Reject ordinary leaves, stems, roots, text, decoration, stains, invented structures, or crops not attached to the plant. Use uncertain for malformed or unreadable cases.'''

def parse_json(raw):
    if not isinstance(raw,str) or not raw.strip(): return None
    s=raw.strip()
    if "```" in s:
        blocks=re.findall(r"```(?:json)?\s*(.*?)```",s,flags=re.I|re.S)
        if blocks:s=blocks[0].strip()
    try:return json.loads(s)
    except Exception:
        for a,b in (("[","]"),("{","}")):
            i=s.find(a);j=s.rfind(b)
            if i>=0 and j>i:
                try:return json.loads(s[i:j+1])
                except Exception:pass
    return None

def proposals_from_raw(raw):
    parsed=parse_json(raw)
    if isinstance(parsed,dict): rows=parsed.get("reproductive_structures") or parsed.get("proposals") or []
    elif isinstance(parsed,list): rows=parsed
    else: rows=[]
    out=[]; rejected=[]
    for pos,z in enumerate(rows[:5]):
        if not isinstance(z,dict):
            rejected.append({"response_position":pos,"reason":"non-object proposal","raw":z});continue
        cls=str(z.get("class","")).lower().strip()
        if cls not in ALLOWED:
            rejected.append({"response_position":pos,"reason":"class outside frozen vocabulary","raw":z});continue
        coords=z.get("bbox_1000")
        alias="bbox_1000"
        if coords is None: coords=z.get("bbox_2d");alias="bbox_2d"
        if coords is None: coords=z.get("bbox");alias="bbox"
        out.append({"response_position":pos,"proposal_index":z.get("proposal_index",pos),"proposed_class":cls,"proposal_confidence":ns["num"](z.get("confidence")),"proposal_description":z.get("description"),"coordinate_alias":alias,"coordinates":coords,"detector_raw":z})
    return out,rejected,parsed

def reviews_from_answer(ans):
    if isinstance(ans,dict): rows=ans.get("reviews") or ans.get("reproductive_structures") or []
    elif isinstance(ans,list): rows=ans
    else:rows=[]
    by={}
    for pos,z in enumerate(rows):
        if isinstance(z,dict):by[str(z.get("proposal_index",pos))]=z
    return by

def load_checkpoint(base_sha):
    try:
        x=get(BRIDGE+PREFIX+"/reproductive_repair_checkpoint.json").json()
        if x.get("base_channel_manifest_sha256")!=base_sha or x.get("parser_revision")!=PARSER_REVISION:raise RuntimeError("checkpoint mismatch")
        return x
    except Exception:return {"protocol_id":PROTOCOL,"corpus_id":CID,"base_channel_manifest_sha256":base_sha,"parser_revision":PARSER_REVISION,"qa_revision":QA_REVISION,"plants":[]}

def save_checkpoint(x):
    upload(PREFIX+"/reproductive_repair_checkpoint.json","application/json",json.dumps(x,indent=2,sort_keys=True).encode())

def main():
    base=get(BRIDGE+PREFIX+"/channels_manifest_frozen.json").json()
    if base.get("protocol_id")!=PROTOCOL:raise RuntimeError("protocol mismatch")
    base_sha=base.get("channel_manifest_sha256")
    cp=load_checkpoint(base_sha); done={p["plant_id"]:p for p in cp.get("plants",[]) if p.get("repair_complete")}
    P=H["AutoProcessor"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),min_pixels=256*28*28,max_pixels=1280*28*28)
    M=H["Qwen2_5_VLForConditionalGeneration"].from_pretrained(MODEL,token=os.environ.get("HF_TOKEN"),torch_dtype=torch.bfloat16,device_map="auto",attn_implementation="sdpa").eval(); qrev=getattr(M.config,"_commit_hash",None) or "unknown"
    repaired=[]
    for n,p in enumerate(base.get("plants",[]),1):
        pid=p.get("plant_id")
        if pid in done:
            repaired.append(done[pid]);continue
        rec=dict(p); rec["repro_parser_revision"]=PARSER_REVISION;rec["repro_qa_revision"]=QA_REVISION;rec["repro_qa_model_revision"]=qrev
        rec["reproductive_structures"]=[];rec["reproductive_parser_rejections"]=[]
        props,rejections,parsed=proposals_from_raw(p.get("reproductive_response_raw"));rec["reproductive_parser_rejections"].extend(rejections);rec["reproductive_response_parsed"]=parsed
        images=[]; candidates=[]
        if p.get("mask_valid") and p.get("masked_crop_path"):
            full=Image.open(io.BytesIO(get(BRIDGE+p["masked_crop_path"]).content)).convert("RGB");images=[full]
            for z in props:
                rb=box(z.get("coordinates"),full.width,full.height,14)
                if not rb:
                    rec["reproductive_parser_rejections"].append({**z,"reason":"invalid normalized box"});continue
                rcb=expand(rb,full.width,full.height,.12,.12,.12,.12);rim=full.crop(rcb);rid=f"{pid}_mr{len(candidates):02d}";data=png(rim);path=f"{PREFIX}/masked_reproductive/{rid}.png";upload(path,"image/png",data)
                candidates.append({**z,"repro_id":rid,"bbox":bd(rb),"crop_bbox":bd(rcb),"crop_path":path,"crop_sha256":sha(data)});images.append(rim)
        if candidates:
            ans,raw=infer(M,P,images,BATCH_QA,1000);reviews=reviews_from_answer(ans);rec["reproductive_batch_qa_raw"]=ans;rec["reproductive_batch_qa_response_raw"]=raw
            for pos,z in enumerate(candidates):
                review=reviews.get(str(z.get("proposal_index"))) or reviews.get(str(pos))
                st,cl,cf,reason=rqa(review)
                if isinstance(review,dict) and review.get("attached_to_plant") is False:st="reject";cl="none";reason=(reason+"; not attached to plant").strip("; ")
                rec["reproductive_structures"].append({**z,"qa_status":st,"qa_class":cl,"qa_confidence":cf,"qa_reason":reason,"qa_raw":review})
        rec["repair_complete"]=True;repaired.append(rec);cp["plants"]=repaired+[done[k] for k in done if k not in {x["plant_id"] for x in repaired}]
        if n%5==0:save_checkpoint(cp)
        print(json.dumps({"event":"repro_repair","corpus":CID,"n":n,"total":len(base.get('plants',[])),"plant":pid,"proposals":len(candidates),"accept":sum(x.get('qa_status')=='accept' for x in rec['reproductive_structures']),"partial":sum(x.get('qa_status')=='partial' for x in rec['reproductive_structures'])},sort_keys=True),flush=True)
    out=dict(base);out["plants"]=repaired;out["reproductive_parser_revision"]=PARSER_REVISION;out["reproductive_qa_revision"]=QA_REVISION;out["reproductive_qa_model"]=MODEL;out["reproductive_qa_model_revision"]=qrev
    old=out.get("counts",{});out["counts"]={**old,"reproductive_proposals":sum(len(p.get("reproductive_structures",[])) for p in repaired),"repro_accept":sum(z.get("qa_status")=="accept" for p in repaired for z in p.get("reproductive_structures",[])),"repro_partial":sum(z.get("qa_status")=="partial" for p in repaired for z in p.get("reproductive_structures",[])),"repro_reject":sum(z.get("qa_status")=="reject" for p in repaired for z in p.get("reproductive_structures",[])),"repro_uncertain":sum(z.get("qa_status")=="uncertain" for p in repaired for z in p.get("reproductive_structures",[])),"parser_rejections":sum(len(p.get("reproductive_parser_rejections",[])) for p in repaired)}
    out["pre_repair_channel_manifest_sha256"]=base_sha;out.pop("channel_manifest_sha256",None);out.pop("frozen_at_utc",None);out["channel_manifest_sha256"]=csha(out);out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());upload(PREFIX+"/channels_manifest_frozen.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode());upload(PREFIX+"/channels_report.json","application/json",json.dumps({k:out.get(k) for k in ("protocol_id","corpus_id","whole_manifest_sha256","color_mask_manifest_sha256","pre_repair_channel_manifest_sha256","channel_manifest_sha256","reproductive_parser_revision","reproductive_qa_revision","counts")},indent=2,sort_keys=True).encode());cp["plants"]=repaired;cp["complete"]=True;save_checkpoint(cp)
    print("RESULT_JSON="+json.dumps({"corpus_id":CID,"channel_manifest_sha256":out["channel_manifest_sha256"],"counts":out["counts"]},sort_keys=True),flush=True)
    del M,P;gc.collect();torch.cuda.empty_cache()
if __name__=="__main__":main()
