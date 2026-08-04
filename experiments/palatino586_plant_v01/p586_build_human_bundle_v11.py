#!/usr/bin/env python3
"""Build the complete evidence ZIP and upload using Supabase signed TUS."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/3aa5ddd6e9a298ab2fb600993bcf044959dc617e/experiments/palatino586_plant_v01/p586_build_human_bundle_v4.py"
source=requests.get(URL,timeout=120).text
old='''        ticket=S.post(TICKET,headers={"x-upload-token":TOKEN},json={},timeout=120);ticket.raise_for_status();info=ticket.json();url=info.get("signedUrl") or info.get("signedURL");url=url if url.startswith("http") else SUPA+url
        with open(out,"rb") as f:
            up=S.put(url,headers={"content-type":"application/zip","x-upsert":"true"},data=f,timeout=3600);up.raise_for_status()
        meta={"protocol_id":PROTOCOL,"path":info["path"],"bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
'''
new='''        ticket=S.post(TICKET,headers={"x-upload-token":TOKEN},json={},timeout=120);ticket.raise_for_status();info=ticket.json();signature=info["token"]
        tus_endpoint="https://ymaqlcfjmdwncdbjprmw.storage.supabase.co/storage/v1/upload/resumable/sign"
        import base64 as _b64
        def _m(v):return _b64.b64encode(v.encode()).decode()
        metadata=",".join([f"bucketName {_m('bridge')}",f"objectName {_m(info['path'])}",f"contentType {_m('application/zip')}",f"cacheControl {_m('3600')}"])
        common={"Tus-Resumable":"1.0.0","x-signature":signature,"x-upsert":"true"}
        start=S.post(tus_endpoint,headers={**common,"Upload-Length":str(size),"Upload-Metadata":metadata},data=b"",timeout=180);start.raise_for_status();upload_url=start.headers.get("Location") or start.headers.get("location")
        if not upload_url:raise RuntimeError("signed TUS creation returned no Location")
        if not upload_url.startswith("http"):upload_url="https://ymaqlcfjmdwncdbjprmw.storage.supabase.co"+upload_url
        offset=int(start.headers.get("Upload-Offset","0"));chunk=6*1024*1024
        with open(out,"rb") as f:
            f.seek(offset)
            while offset<size:
                data=f.read(min(chunk,size-offset))
                r=S.patch(upload_url,headers={**common,"Upload-Offset":str(offset),"Content-Type":"application/offset+octet-stream"},data=data,timeout=600);r.raise_for_status();nxt=int(r.headers.get("Upload-Offset",str(offset+len(data))))
                if nxt!=offset+len(data):raise RuntimeError(f"TUS offset mismatch {offset}+{len(data)} != {nxt}")
                offset=nxt
                if offset==size or offset%(60*1024*1024)<chunk:print(json.dumps({"event":"bundle_upload_progress","bytes":offset,"total":size,"percent":round(100*offset/size,2)},sort_keys=True),flush=True)
        meta={"protocol_id":PROTOCOL,"path":info["path"],"public_url":BRIDGE+info["path"],"bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass","upload_transport":"Supabase signed TUS /resumable/sign","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
'''
if source.count(old)!=1:raise RuntimeError(f"signed TUS patch point mismatch: {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
