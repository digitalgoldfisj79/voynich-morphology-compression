#!/usr/bin/env python3
"""Resume the bounded control extractor from its persisted checkpoint.

This changes only execution persistence. Page order, screen threshold, object cap,
prompts and QA rules remain identical.
"""
import requests,time
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/28c9e58b79cb269bf9e630336d78d8854883738d/experiments/palatino586_plant_v01/p586_control_extract_bounded.py"
source=requests.get(URL,timeout=120).text
old='''    out={"protocol_id":"P586-VMS-PLANT-0.1-20260803","amendment":"PREREGISTRATION_AMENDMENT_01","manuscript_id":CID,"page_registry_count":len(rows),"detector_model":MODEL,"detector_revision":rev,"screen_threshold":.70,"broad_cap":20,"pages":[],"plants":[]}\n    for pi,row in enumerate(rows,1):'''
new='''    checkpoint_url=f"https://ymaqlcfjmdwncdbjprmw.supabase.co/storage/v1/object/public/bridge/{PREFIX}/checkpoint.json"\n    try:\n        prior=requests.get(checkpoint_url,timeout=60); prior.raise_for_status(); out=prior.json()\n        if out.get("manuscript_id")!=CID or out.get("protocol_id")!="P586-VMS-PLANT-0.1-20260803": raise RuntimeError("checkpoint identity mismatch")\n    except Exception:\n        out={"protocol_id":"P586-VMS-PLANT-0.1-20260803","amendment":"PREREGISTRATION_AMENDMENT_01","manuscript_id":CID,"page_registry_count":len(rows),"detector_model":MODEL,"detector_revision":rev,"screen_threshold":.70,"broad_cap":20,"pages":[],"plants":[]}\n    start=len(out.get("pages",[]))\n    for pi,row in enumerate(rows[start:],start+1):'''
if source.count(old)!=1: raise RuntimeError(f"resume patch expected once, found {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__","requests":requests,"time":time})
