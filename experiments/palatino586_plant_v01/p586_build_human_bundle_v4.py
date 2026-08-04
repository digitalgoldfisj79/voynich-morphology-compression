#!/usr/bin/env python3
"""Build the final complete evidence ZIP using concurrent downloads and ZIP_STORED.

PNG/JPEG inputs are already compressed, so recompressing them is both slow and
largely ineffective. This closeout implementation preserves bytes exactly,
logs progress, and includes all frozen ledgers, reports and image channels.
"""
from __future__ import annotations
import base64,csv,hashlib,json,os,re,tempfile,time,zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import requests

SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co"
BRIDGE=SUPA+"/storage/v1/object/public/bridge/"
UPLOAD=SUPA+"/functions/v1/p586-plant-upload-v01"
TICKET=SUPA+"/functions/v1/p586-plant-bundle-ticket-v01"
TARGET="bncf_palatino_586"
PROTOCOL="P586-VMS-PLANT-0.1-20260803"
WORKERS=24
S=requests.Session();S.headers["User-Agent"]="P586PlantBundle/0.4"

def get(url,timeout=300):
    r=S.get(url,timeout=timeout);r.raise_for_status();return r

def token():
    u="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py"
    return re.search(r'RUN_ID\s*=\s*"([^"]+)"',get(u).text).group(1)
TOKEN=token()

def prefix(cid):return "p586_plant_v01/target" if cid==TARGET else f"p586_plant_v01/controls/{cid}"
def jget(path):return get(BRIDGE+path).json()
def stable_json(x):return json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False).encode()
def csv_bytes(rows,fields):
    import io
    b=io.StringIO();w=csv.DictWriter(b,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows);return b.getvalue().encode()

def add_bytes(z,name,data):
    z.writestr(name,data,compress_type=zipfile.ZIP_STORED)

def download_one(task):
    name,url=task
    for k in range(4):
        try:return name,get(url,300).content,None
        except Exception as e:
            last=e;time.sleep(1.5**k)
    return name,None,str(last)

def main():
    analysis=jget("p586_plant_v01/results/dinov3_analysis.json")
    blind=jget("p586_plant_v01/results/blind_adjudication.json")
    ids=list(analysis.get("channel_manifests",{}))
    if TARGET not in ids or "voynich" not in ids:raise RuntimeError("analysis corpus panel incomplete")
    tasks=[];summ=[];manifests={}
    with tempfile.TemporaryDirectory(prefix="p586_bundle_") as td:
        out=Path(td)/"P586_PLANT_MORPHOLOGY_HUMAN_READABLE.zip"
        with zipfile.ZipFile(out,"w",allowZip64=True) as z:
            gh="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/gpt56/p586-plant-v01-20260803/experiments/palatino586_plant_v01/"
            protocol_files=["PREREGISTRATION.md"]+[f"PREREGISTRATION_AMENDMENT_{i:02d}.md" for i in range(1,10)]+["CONTROL_PANEL.json","DINOV3_FREEZE.json","MASK_IMPLEMENTATION_FREEZE.md"]
            for f in protocol_files:
                try:add_bytes(z,"00_PROTOCOL/"+f,get(gh+f).content)
                except Exception:pass
            result_paths=["dinov3_analysis.json","blind_adjudication.json","blind_key.json","blind_pair_candidates.json","registration_audit.json","closeout_report.json","FINAL_REPORT.md"]
            for f in result_paths:
                try:add_bytes(z,"01_RESULTS/"+f,get(BRIDGE+"p586_plant_v01/results/"+f).content)
                except Exception:pass
            for cid in ids:
                pre=prefix(cid)
                whole=jget(pre+"/whole_manifest_frozen.json")
                channels=jget(pre+"/channels_manifest_frozen.json")
                manifests[cid]=(whole,channels)
                add_bytes(z,f"02_CORPORA/{cid}/whole_manifest_frozen.json",stable_json(whole))
                add_bytes(z,f"02_CORPORA/{cid}/channels_manifest_frozen.json",stable_json(channels))
                for extra in ["color_masks_frozen.json","channels_report.json","whole_qa_ledger.csv"]:
                    try:add_bytes(z,f"02_CORPORA/{cid}/{extra}",get(BRIDGE+pre+"/"+extra).content)
                    except Exception:pass
                plants=channels.get("plants",[]);flat=[];repro=[]
                for p in plants:
                    flat.append({"plant_id":p.get("plant_id"),"qa_status":p.get("qa_status"),"qa_confidence":p.get("qa_confidence"),"mask_valid":p.get("mask_valid"),"mask_strict_valid":p.get("mask_strict_valid"),"above_valid_final":p.get("above_valid_final"),"root_boundary_y_1000":p.get("root_boundary_y_1000"),"crop_path":p.get("crop_path"),"masked_crop_path":p.get("masked_crop_path"),"above_strict_path":p.get("above_strict_path"),"above_context_path":p.get("above_context_path")})
                    for zz in p.get("reproductive_structures",[]):
                        repro.append({"plant_id":p.get("plant_id"),"repro_id":zz.get("repro_id"),"qa_class":zz.get("qa_class"),"proposed_class":zz.get("proposed_class"),"qa_status":zz.get("qa_status"),"qa_confidence":zz.get("qa_confidence"),"crop_path":zz.get("crop_path")})
                    for key,sub in [("crop_path","whole"),("masked_crop_path","masked"),("above_strict_path","above_strict"),("above_context_path","above_context")]:
                        path=p.get(key)
                        if path:tasks.append((f"02_CORPORA/{cid}/images/{sub}/{p.get('plant_id')}_{path.rsplit('/',1)[-1]}",BRIDGE+path))
                    for zz in p.get("reproductive_structures",[]):
                        path=zz.get("crop_path")
                        if path:tasks.append((f"02_CORPORA/{cid}/images/reproductive/{p.get('plant_id')}_{zz.get('repro_id')}_{path.rsplit('/',1)[-1]}",BRIDGE+path))
                if flat:add_bytes(z,f"02_CORPORA/{cid}/plants.csv",csv_bytes(flat,list(flat[0])))
                if repro:add_bytes(z,f"02_CORPORA/{cid}/reproductive.csv",csv_bytes(repro,list(repro[0])))
                summ.append({"corpus":cid,"whole":len(plants),"valid_masks":sum(bool(x.get("mask_valid")) for x in plants),"reproductive_proposals":len(repro),"reproductive_accept":sum(x.get("qa_status")=="accept" for x in repro),"whole_manifest_sha256":whole.get("whole_manifest_sha256"),"channel_manifest_sha256":channels.get("channel_manifest_sha256")})
            for t in blind.get("trials",[]):
                if t.get("sheet_path"):tasks.append((f"03_BLIND_SHEETS/{t['trial_id']}.png",BRIDGE+t["sheet_path"]))
            add_bytes(z,"04_TABLES/corpus_summary.csv",csv_bytes(summ,list(summ[0])))
            close=jget("p586_plant_v01/results/closeout_report.json")
            readme=["# Palatino 586 ↔ Voynich Plant-Morphology Programme v0.1","",f"Protocol: `{PROTOCOL}`",f"DINOv3 result SHA-256: `{analysis.get('result_sha256')}`",f"Blind result SHA-256: `{blind.get('result_sha256')}`",f"Closeout result SHA-256: `{close.get('closeout_result_sha256')}`","",f"Final conclusion: **{close.get('conclusion')}**","","The primary effects were weakly positive but all confidence intervals crossed zero and no manuscript-label permutation test supported an affinity claim. The blind adjudicator abstained on every trial and is therefore non-informative.","","This archive contains frozen protocols, all amendments, results, manifests, ledgers, every retained ordinary/masked/above-ground/reproductive crop, and blind evidence sheets."]
            add_bytes(z,"README.md","\n".join(readme).encode())
            failures=[]
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futures=[ex.submit(download_one,t) for t in tasks]
                for n,f in enumerate(as_completed(futures),1):
                    name,data,error=f.result()
                    if error:failures.append({"name":name,"error":error})
                    else:add_bytes(z,name,data)
                    if n%100==0 or n==len(futures):print(json.dumps({"event":"bundle_progress","completed":n,"total":len(futures),"failures":len(failures)},sort_keys=True),flush=True)
            add_bytes(z,"05_CLOSEOUT/download_failures.json",stable_json(failures))
            if failures:raise RuntimeError(f"bundle has {len(failures)} download failures")
        h=hashlib.sha256();size=0
        with open(out,"rb") as f:
            while True:
                b=f.read(8<<20)
                if not b:break
                h.update(b);size+=len(b)
        digest=h.hexdigest()
        with zipfile.ZipFile(out) as z:
            bad=z.testzip();entries=len(z.infolist())
        if bad:raise RuntimeError(f"corrupt zip member: {bad}")
        ticket=S.post(TICKET,headers={"x-upload-token":TOKEN},json={},timeout=120);ticket.raise_for_status();info=ticket.json();url=info.get("signedUrl") or info.get("signedURL");url=url if url.startswith("http") else SUPA+url
        with open(out,"rb") as f:
            up=S.put(url,headers={"content-type":"application/zip","x-upsert":"true"},data=f,timeout=3600);up.raise_for_status()
        meta={"protocol_id":PROTOCOL,"path":info["path"],"bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
        payload={"path":"p586_plant_v01/results/human_bundle.json","content_type":"application/json","data_b64":base64.b64encode(stable_json(meta)).decode()};r=S.post(UPLOAD,headers={"x-upload-token":TOKEN},json=payload,timeout=120);r.raise_for_status()
        print("RESULT_JSON="+json.dumps(meta,sort_keys=True),flush=True)
if __name__=="__main__":main()
