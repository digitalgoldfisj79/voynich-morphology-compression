#!/usr/bin/env python3
"""Run the concurrent evidence builder with authenticated 6 MiB TUS upload."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/3aa5ddd6e9a298ab2fb600993bcf044959dc617e/experiments/palatino586_plant_v01/p586_build_human_bundle_v4.py"
source=requests.get(URL,timeout=120).text
old='''        ticket=S.post(TICKET,headers={"x-upload-token":TOKEN},json={},timeout=120);ticket.raise_for_status();info=ticket.json();url=info.get("signedUrl") or info.get("signedURL");url=url if url.startswith("http") else SUPA+url
        with open(out,"rb") as f:
            up=S.put(url,headers={"content-type":"application/zip","x-upsert":"true"},data=f,timeout=3600);up.raise_for_status()
        meta={"protocol_id":PROTOCOL,"path":info["path"],"bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
'''
new='''        final_path="p586_plant_v01/P586_PLANT_MORPHOLOGY_HUMAN_READABLE.zip"
        proxy=SUPA+"/functions/v1/p586-bundle-tus-v01"
        start=S.post(proxy,headers={"x-p586-token":TOKEN,"x-action":"start"},json={"path":final_path,"size":size,"content_type":"application/zip"},timeout=180);start.raise_for_status();state=start.json();upload_url=state["upload_url"];offset=int(state.get("offset",0));chunk=6*1024*1024
        with open(out,"rb") as f:
            f.seek(offset)
            while offset<size:
                data=f.read(min(chunk,size-offset))
                r=S.post(proxy,headers={"x-p586-token":TOKEN,"x-action":"patch","x-upload-url":upload_url,"x-upload-offset":str(offset),"content-type":"application/octet-stream"},data=data,timeout=600);r.raise_for_status();nxt=int(r.json()["offset"])
                if nxt!=offset+len(data):raise RuntimeError(f"TUS offset mismatch {offset}+{len(data)} != {nxt}")
                offset=nxt
                if offset==size or offset%(60*1024*1024)<chunk:print(json.dumps({"event":"bundle_upload_progress","bytes":offset,"total":size,"percent":round(100*offset/size,2)},sort_keys=True),flush=True)
        meta={"protocol_id":PROTOCOL,"path":final_path,"bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass","upload_transport":"TUS via retired-after-use authenticated proxy","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
'''
if source.count(old)!=1:raise RuntimeError(f"upload patch point mismatch: {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
