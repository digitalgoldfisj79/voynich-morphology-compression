#!/usr/bin/env python3
"""Merge Amendment-09 target repair checkpoints and shard result into one frozen manifest."""
from __future__ import annotations
import base64,hashlib,json,re,time,requests
SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co";BRIDGE=SUPA+"/storage/v1/object/public/bridge/";UPLOAD=SUPA+"/functions/v1/p586-plant-upload-v01";PREFIX="p586_plant_v01/target";PROTOCOL="P586-VMS-PLANT-0.1-20260803"
S=requests.Session();S.headers["User-Agent"]="P586ReproMerge/0.1"
def get(path):r=S.get(BRIDGE+path,timeout=300);r.raise_for_status();return r.json()
def token():
 s=requests.get("https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py",timeout=120).text;return re.search(r'RUN_ID\s*=\s*"([^"]+)"',s).group(1)
T=token()
def upload(path,obj):
 data=json.dumps(obj,indent=2,sort_keys=True).encode();r=S.post(UPLOAD,headers={"x-upload-token":T},json={"path":path,"content_type":"application/json","data_b64":base64.b64encode(data).decode()},timeout=300);r.raise_for_status()
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def main():
 base=get(PREFIX+"/channels_manifest_frozen.json");cp=get(PREFIX+"/reproductive_repair_checkpoint.json");shard=get(PREFIX+"/reproductive_shards/result_100_194.json")
 if base.get("protocol_id")!=PROTOCOL or cp.get("protocol_id")!=PROTOCOL or shard.get("protocol_id")!=PROTOCOL:raise RuntimeError("protocol mismatch")
 base_sha=base.get("channel_manifest_sha256")
 if cp.get("base_channel_manifest_sha256")!=base_sha or shard.get("pre_repair_channel_manifest_sha256")!=base_sha:raise RuntimeError("base hash mismatch")
 order=[p.get("plant_id") for p in base.get("plants",[])];a=order[:100];b=order[100:194];am={p.get("plant_id"):p for p in cp.get("plants",[]) if p.get("repair_complete") and p.get("plant_id") in set(a)};bm={p.get("plant_id"):p for p in shard.get("plants",[]) if p.get("repair_complete") and p.get("plant_id") in set(b)}
 if set(am)!=set(a):raise RuntimeError(f"shard A incomplete: {len(am)}/{len(a)}")
 if set(bm)!=set(b):raise RuntimeError(f"shard B incomplete: {len(bm)}/{len(b)}")
 plants=[am[x] if i<100 else bm[x] for i,x in enumerate(order)]
 out=dict(base);out["plants"]=plants;out["amendment_09_shards"]={"A":[0,100],"B":[100,194],"base_channel_manifest_sha256":base_sha};out["reproductive_parser_revision"]="repro-interface-v2-list-bbox2d";out["reproductive_qa_revision"]="repro-batch-qa-v1";out["reproductive_qa_model"]="Qwen/Qwen2.5-VL-7B-Instruct";out["reproductive_qa_model_revision"]=shard.get("reproductive_qa_model_revision") or cp.get("reproductive_qa_model_revision")
 old=out.get("counts",{});out["counts"]={**old,"reproductive_proposals":sum(len(p.get("reproductive_structures",[])) for p in plants),"repro_accept":sum(z.get("qa_status")=="accept" for p in plants for z in p.get("reproductive_structures",[])),"repro_partial":sum(z.get("qa_status")=="partial" for p in plants for z in p.get("reproductive_structures",[])),"repro_reject":sum(z.get("qa_status")=="reject" for p in plants for z in p.get("reproductive_structures",[])),"repro_uncertain":sum(z.get("qa_status")=="uncertain" for p in plants for z in p.get("reproductive_structures",[])),"parser_rejections":sum(len(p.get("reproductive_parser_rejections",[])) for p in plants)}
 out["pre_repair_channel_manifest_sha256"]=base_sha;out.pop("channel_manifest_sha256",None);out.pop("frozen_at_utc",None);out["channel_manifest_sha256"]=csha(out);out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());upload(PREFIX+"/channels_manifest_frozen.json",out);upload(PREFIX+"/channels_report.json",{k:out.get(k) for k in ("protocol_id","corpus_id","whole_manifest_sha256","color_mask_manifest_sha256","pre_repair_channel_manifest_sha256","channel_manifest_sha256","reproductive_parser_revision","reproductive_qa_revision","amendment_09_shards","counts")});print("RESULT_JSON="+json.dumps({"channel_manifest_sha256":out["channel_manifest_sha256"],"counts":out["counts"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
