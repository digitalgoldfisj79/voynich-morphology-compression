#!/usr/bin/env python3
"""Build the complete evidence ZIP and hand it off through a private HF dataset."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/3aa5ddd6e9a298ab2fb600993bcf044959dc617e/experiments/palatino586_plant_v01/p586_build_human_bundle_v4.py"
source=requests.get(URL,timeout=120).text
old='''        ticket=S.post(TICKET,headers={"x-upload-token":TOKEN},json={},timeout=120);ticket.raise_for_status();info=ticket.json();url=info.get("signedUrl") or info.get("signedURL");url=url if url.startswith("http") else SUPA+url
        with open(out,"rb") as f:
            up=S.put(url,headers={"content-type":"application/zip","x-upsert":"true"},data=f,timeout=3600);up.raise_for_status()
        meta={"protocol_id":PROTOCOL,"path":info["path"],"bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
'''
new='''        from huggingface_hub import HfApi
        repo_id="Digitalgoldfish79/p586-plant-morphology-artifact-v01"
        path_in_repo="P586_PLANT_MORPHOLOGY_COMPLETE.zip"
        api=HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo_id=repo_id,repo_type="dataset",private=True,exist_ok=True)
        commit=api.upload_file(path_or_fileobj=str(out),path_in_repo=path_in_repo,repo_id=repo_id,repo_type="dataset",commit_message="Upload completed Palatino 586 plant-morphology evidence bundle")
        meta={"protocol_id":PROTOCOL,"path":f"hf://datasets/{repo_id}/{path_in_repo}","temporary_private_repo":repo_id,"path_in_repo":path_in_repo,"hf_commit_oid":getattr(commit,"oid",None),"bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
        api.upload_file(path_or_fileobj=stable_json(meta),path_in_repo="artifact_manifest.json",repo_id=repo_id,repo_type="dataset",commit_message="Add artifact integrity manifest")
'''
if source.count(old)!=1:raise RuntimeError(f"handoff patch point mismatch: {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
