#!/usr/bin/env python3
"""Finalize Amendment-08 secondary controls from frozen whole and colour manifests."""
from __future__ import annotations
import base64,hashlib,json,os,re,time,requests
PROTOCOL="P586-VMS-PLANT-0.1-20260803";CID=os.environ["CORPUS_ID"]
SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co";BRIDGE=SUPA+"/storage/v1/object/public/bridge/";EP=SUPA+"/functions/v1/p586-plant-upload-v01";PREFIX=f"p586_plant_v01/controls/{CID}"
S=requests.Session();S.headers["User-Agent"]="P586SecondaryFinalize/0.1"
def get(u):r=S.get(u,timeout=240);r.raise_for_status();return r
def token():
 s=get("https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py").text;return re.search(r'RUN_ID\s*=\s*"([^"]+)"',s).group(1)
TOKEN=token()
def upload(path,typ,data):
 r=S.post(EP,headers={"x-upload-token":TOKEN},json={"path":path,"content_type":typ,"data_b64":base64.b64encode(data).decode()},timeout=360);r.raise_for_status()
def csha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def main():
 whole=get(BRIDGE+PREFIX+"/whole_manifest_frozen.json").json();colors=get(BRIDGE+PREFIX+"/color_masks_frozen.json").json();broad=[p for p in whole.get("plants",[]) if p.get("qa_status") in {"accept","partial"}];cmap={r.get("plant_id"):r for r in colors.get("records",[])}
 if len(cmap)!=len(broad) or set(cmap)!={p.get("plant_id") for p in broad}:raise RuntimeError("colour ledger is not terminal for every broad plant")
 success=sum(r.get("status")=="success" and r.get("area_valid") for r in cmap.values())
 if success<8:raise RuntimeError(f"secondary control has only {success} valid masks")
 plants=[]
 for p in broad:
  m=cmap[p["plant_id"]];valid=bool(m.get("status")=="success" and m.get("area_valid"));q={**p,"mask_status":m.get("status"),"mask_model":m.get("model"),"mask_prompt_sha256":m.get("prompt_sha256"),"raw_mask_path":m.get("raw_mask_path"),"raw_mask_sha256":m.get("raw_mask_sha256"),"quantized_mask_path":m.get("quantized_mask_path"),"quantized_mask_sha256":m.get("quantized_mask_sha256"),"mask_resized":m.get("mask_resized"),"green_fraction":m.get("green_fraction"),"red_fraction":m.get("red_fraction"),"foreground_fraction":m.get("foreground_fraction"),"mean_nearest_colour_distance":m.get("mean_nearest_colour_distance"),"p95_nearest_colour_distance":m.get("p95_nearest_colour_distance"),"area_valid":bool(m.get("area_valid")),"masked_crop_path":m.get("masked_crop_path"),"masked_crop_sha256":m.get("masked_crop_sha256"),"above_strict_path":m.get("above_strict_path"),"above_strict_sha256":m.get("above_strict_sha256"),"above_context_path":m.get("above_context_path"),"above_context_sha256":m.get("above_context_sha256"),"mask_audit_required":not bool(m.get("area_valid")),"mask_audit_status":"not_auditable" if m.get("status")!="success" else "objective_gate","mask_valid":valid,"mask_strict_valid":valid and p.get("qa_status")=="accept","above_valid_final":valid and bool(m.get("above_valid")),"reproductive_structures":[],"reproductive_scope":"not_run_secondary_control_amendment_08"};plants.append(q)
 out={"protocol_id":PROTOCOL,"corpus_id":CID,"amendments":["PREREGISTRATION_AMENDMENT_04","PREREGISTRATION_AMENDMENT_07","PREREGISTRATION_AMENDMENT_08"],"whole_manifest_sha256":whole.get("whole_manifest_sha256"),"color_mask_manifest_sha256":colors.get("color_mask_manifest_sha256"),"secondary_control":True,"reproductive_scope":"not_run","plants":plants}
 out["counts"]={"plants":len(plants),"mask_success":sum(p.get("mask_status")=="success" for p in plants),"area_valid":sum(p.get("area_valid") for p in plants),"mask_broad_valid":sum(p.get("mask_valid") for p in plants),"mask_strict_valid":sum(p.get("mask_strict_valid") for p in plants),"above_broad_valid":sum(p.get("above_valid_final") for p in plants),"terminal_failures":sum(p.get("mask_status")!="success" for p in plants),"reproductive_proposals":0,"repro_accept":0,"repro_partial":0};out["channel_manifest_sha256"]=csha(out);out["frozen_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());upload(PREFIX+"/channels_manifest_frozen.json","application/json",json.dumps(out,indent=2,sort_keys=True).encode());upload(PREFIX+"/channels_report.json","application/json",json.dumps({k:out[k] for k in ("protocol_id","corpus_id","whole_manifest_sha256","color_mask_manifest_sha256","channel_manifest_sha256","counts")},indent=2,sort_keys=True).encode());print("RESULT_JSON="+json.dumps({"corpus_id":CID,"channel_manifest_sha256":out["channel_manifest_sha256"],"counts":out["counts"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
