#!/usr/bin/env python3
"""Build the deterministic cross-channel decision report after DINO, blind review and registration."""
from __future__ import annotations
import base64,hashlib,json,re,time,requests
SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co";BRIDGE=SUPA+"/storage/v1/object/public/bridge/";UPLOAD=SUPA+"/functions/v1/p586-plant-upload-v01";PROTOCOL="P586-VMS-PLANT-0.1-20260803"
S=requests.Session();S.headers["User-Agent"]="P586Closeout/0.1"
def get(path):r=S.get(BRIDGE+path,timeout=300);r.raise_for_status();return r
def token():
 r=requests.get("https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py",timeout=120);r.raise_for_status();return re.search(r'RUN_ID\s*=\s*"([^"]+)"',r.text).group(1)
T=token()
def upload(path,typ,data):
 r=S.post(UPLOAD,headers={"x-upload-token":T},json={"path":path,"content_type":typ,"data_b64":base64.b64encode(data).decode()},timeout=240);r.raise_for_status()
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def score(v):
 e=v.get("primary_effect_target_minus_control_mean");ci=v.get("hierarchical_bootstrap_95ci") or [None,None];p=v.get("manuscript_label_rank_p_upper")
 return {"effect":e,"ci":ci,"p":p,"positive":e is not None and e>0,"ci_excludes_zero_positive":len(ci)==2 and ci[0] is not None and ci[0]>0,"permutation_support":p is not None and p<=0.05,"eligible_controls":len(v.get("eligible_controls") or [])}
def main():
 analysis=get("p586_plant_v01/results/dinov3_analysis.json").json();blind=get("p586_plant_v01/results/blind_adjudication.json").json();registration=get("p586_plant_v01/results/registration_audit.json").json()
 primary={"whole":"whole_masked_broad","above":"above_context_broad","flowers":"flowers_broad"};channels={k:score(analysis.get("results",{}).get(v,{})) for k,v in primary.items()}
 supported=[k for k,v in channels.items() if v["positive"] and v["ci_excludes_zero_positive"] and v["permutation_support"]];blind_ok={k:(blind.get("summary",{}).get(k,{}).get("target_selected",0)>=blind.get("summary",{}).get(k,{}).get("control_selected",0)) for k in primary};strong=len(supported)>=2 and any(k in supported for k in ("above","flowers")) and all(blind_ok.get(k,False) for k in supported)
 if strong:conclusion="strong_positive_palatinovoynich_affinity"
 elif any(v["positive"] and v["ci_excludes_zero_positive"] for v in channels.values()):conclusion="limited_or_channel_specific_positive"
 elif all(v["effect"] is not None and v["effect"]<=0 for v in channels.values() if v["effect"] is not None):conclusion="negative_or_null_across_primary_channels"
 else:conclusion="mixed_or_underpowered"
 report={"protocol_id":PROTOCOL,"analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"registration_channel_manifest_sha256":registration.get("channel_manifest_sha256"),"primary_variants":primary,"channels":channels,"blind_no_contradiction":blind_ok,"supported_channels":supported,"strong_affinity_rule_passed":strong,"conclusion":conclusion,"claim_classes":{"exact":["source, crop and manifest hashes","frozen object counts","database registration counts"],"machine_certified":["DINO vector dimensions and L2 norms","deterministic similarity and resampling calculations"],"empirical":["manuscript-level effects, intervals, permutation ranks and nearest-neighbour shares","blinded model adjudication outcomes"],"heuristic":["vision-model plant localisation, mask generation and reproductive visual QA"],"open":["historical transmission or botanical identity implications"]},"created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())};report["closeout_result_sha256"]=sha(report)
 md=["# Palatino 586 ↔ Voynich Plant-Morphology Programme — Final Report","",f"Protocol: `{PROTOCOL}`",f"DINO result SHA-256: `{report['analysis_result_sha256']}`",f"Blind result SHA-256: `{report['blind_result_sha256']}`",f"Closeout SHA-256: `{report['closeout_result_sha256']}`","",f"## Decision\n\n**{conclusion}**","",f"Strong-affinity rule passed: **{strong}**","","## Primary channels",""]
 for k,v in channels.items():md.append(f"- **{k}**: effect `{v['effect']}`; 95% CI `{v['ci']}`; manuscript-label p `{v['p']}`; eligible controls `{v['eligible_controls']}`; blind no-contradiction `{blind_ok.get(k)}`.")
 md += ["","## Claim status","","Exact: source/crop/manifests and registration hashes. Machine-certified: DINO and statistical arithmetic. Empirical: observed manuscript effects and blinded adjudication. Heuristic: vision localisation, masking and reproductive QA. Open: any historical-transmission interpretation.",""]
 upload("p586_plant_v01/results/closeout_report.json","application/json",json.dumps(report,indent=2,sort_keys=True).encode());upload("p586_plant_v01/results/FINAL_REPORT.md","text/markdown","\n".join(md).encode());print("RESULT_JSON="+json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":main()
