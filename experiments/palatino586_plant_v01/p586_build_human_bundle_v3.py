#!/usr/bin/env python3
"""Audited v3 launcher for the final human-readable evidence bundle."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/gpt56/p586-plant-v01-20260803/experiments/palatino586_plant_v01/p586_build_human_bundle.py"
source=requests.get(URL,timeout=120).text
old='TARGET="bncf_palatino_586";IDS=[TARGET,"voynich","bnf_lat_6862","herb_8510642bf2ce","herb_c51d653739d2","herb_0b5263630edb","herb_18f0aa144a2b","herb_eaec4fe75d89","herb_78e2bbc79062","herb_d91d01bd5276","bsb1784","herb_7ce7efc90e6d","herb_fafef9a26da5","herb_205bfb89efbc"]'
new='TARGET="bncf_palatino_586";IDS=[]'
if source.count(old)!=1:raise RuntimeError("static ID patch point mismatch")
source=source.replace(old,new,1)
old='analysis=jget("p586_plant_v01/results/dinov3_analysis.json");blind=jget("p586_plant_v01/results/blind_adjudication.json");buf=io.BytesIO();summary=[]'
new='analysis=jget("p586_plant_v01/results/dinov3_analysis.json");blind=jget("p586_plant_v01/results/blind_adjudication.json");global IDS;IDS=list(analysis.get("channel_manifests",{}));buf=io.BytesIO();summary=[]'
if source.count(old)!=1:raise RuntimeError("dynamic ID patch point mismatch")
source=source.replace(old,new,1)
old='for f in ["PREREGISTRATION.md","PREREGISTRATION_AMENDMENT_01.md","CONTROL_PANEL.json","DINOV3_FREEZE.json"]:add_url(z,"00_PROTOCOL/"+f,gh+f)'
new='for f in ["PREREGISTRATION.md"]+[f"PREREGISTRATION_AMENDMENT_{i:02d}.md" for i in range(1,9)]+["CONTROL_PANEL.json","DINOV3_FREEZE.json"]:\n   try:add_url(z,"00_PROTOCOL/"+f,gh+f)\n   except Exception:pass'
if source.count(old)!=1:raise RuntimeError("protocol-list patch point mismatch")
source=source.replace(old,new,1)
old='try:channels=jget(pre+"/channels_manifest_frozen.json");add_json(z,f"02_CORPORA/{cid}/channels_manifest_frozen.json",channels)\n   except Exception:channels=None'
new='try:channels=jget(pre+"/channels_manifest_frozen.json");add_json(z,f"02_CORPORA/{cid}/channels_manifest_frozen.json",channels)\n   except Exception:channels=None\n   for extra in ["color_masks_frozen.json","channels_report.json","whole_qa_ledger.csv"]:\n    try:add_url(z,f"02_CORPORA/{cid}/{extra}",BRIDGE+pre+"/"+extra)\n    except Exception:pass'
if source.count(old)!=1:raise RuntimeError("corpus-extra patch point mismatch")
source=source.replace(old,new,1)
old='for path in ["p586_plant_v01/results/blind_key.json","p586_plant_v01/results/blind_pair_candidates.json"]:'
new='for path in ["p586_plant_v01/results/blind_key.json","p586_plant_v01/results/blind_pair_candidates.json","p586_plant_v01/results/registration_audit.json","p586_plant_v01/results/closeout_report.json"]:'
if source.count(old)!=1:raise RuntimeError("result-extra patch point mismatch")
source=source.replace(old,new,1)
source=source.replace('The ZIP includes frozen protocols, manifests, ordinary/masked/above-ground/reproductive images, CSV ledgers, evidence sheets and corrected results.','The ZIP includes frozen protocols and all amendments, manifests, terminal failure ledgers, ordinary/masked/above-ground/reproductive images, CSV ledgers, evidence sheets, registration audit, endpoint closeout state and corrected results.')
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
