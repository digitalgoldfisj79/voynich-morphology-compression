#!/usr/bin/env python3
"""Register the frozen Palatino plant corpus into Plant Mask Painter and audit it."""
from __future__ import annotations
import base64,json,re,time,requests
SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co";REG=SUPA+"/functions/v1/p586-plant-register-v01";UPLOAD=SUPA+"/functions/v1/p586-plant-upload-v01";BRIDGE=SUPA+"/storage/v1/object/public/bridge/";PROTOCOL="P586-VMS-PLANT-0.1-20260803";MID="bncf_palatino_586"
S=requests.Session();S.headers["User-Agent"]="P586PlantRegisterAudit/0.1"
def get(u):r=S.get(u,timeout=240);r.raise_for_status();return r
def token():
 s=get("https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py").text;return re.search(r'RUN_ID\s*=\s*"([^"]+)"',s).group(1)
T=token()
def upload(path,obj):
 data=json.dumps(obj,indent=2,sort_keys=True).encode();r=S.post(UPLOAD,headers={"x-upload-token":T},json={"path":path,"content_type":"application/json","data_b64":base64.b64encode(data).decode()},timeout=240);r.raise_for_status()
def main():
 status=get(REG).json();offset=0;batches=[]
 while True:
  r=S.post(REG,headers={"x-p586-token":T,"content-type":"application/json"},json={"offset":offset,"limit":25},timeout=1800);r.raise_for_status();x=r.json();batches.append(x)
  if x.get("complete"):break
  nxt=int(x.get("next_offset",offset));
  if nxt<=offset:raise RuntimeError("registration made no progress")
  offset=nxt
  time.sleep(.3)
 whole=get(BRIDGE+"p586_plant_v01/target/whole_manifest_frozen.json").json();channels=get(BRIDGE+"p586_plant_v01/target/channels_manifest_frozen.json").json();expected=sum(p.get("qa_status") in {"accept","partial"} for p in channels.get("plants",[]))
 audit={"protocol_id":PROTOCOL,"manuscript_id":MID,"registration_endpoint_status":status,"channel_manifest_sha256":channels.get("channel_manifest_sha256"),"whole_manifest_sha256":whole.get("whole_manifest_sha256"),"expected_broad_plants":expected,"batches":batches,"registered_reported":sum(int(b.get("processed",0)) for b in batches),"complete":bool(batches and batches[-1].get("complete")),"audited_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
 if audit["registered_reported"]!=expected or not audit["complete"]:raise RuntimeError(f"registration count mismatch: {audit}")
 upload("p586_plant_v01/results/registration_audit.json",audit);print("RESULT_JSON="+json.dumps(audit,sort_keys=True),flush=True)
if __name__=="__main__":main()
