#!/usr/bin/env python3
"""Build persistent core and multipart Palatino 586 evidence bundles."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/3aa5ddd6e9a298ab2fb600993bcf044959dc617e/experiments/palatino586_plant_v01/p586_build_human_bundle_v4.py"
source=requests.get(URL,timeout=120).text
old='''        ticket=S.post(TICKET,headers={"x-upload-token":TOKEN},json={},timeout=120);ticket.raise_for_status();info=ticket.json();url=info.get("signedUrl") or info.get("signedURL");url=url if url.startswith("http") else SUPA+url
        with open(out,"rb") as f:
            up=S.put(url,headers={"content-type":"application/zip","x-upsert":"true"},data=f,timeout=3600);up.raise_for_status()
        meta={"protocol_id":PROTOCOL,"path":info["path"],"bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
'''
new='''        import base64 as _b64
        ticket_ep=SUPA+"/functions/v1/p586-bundle-parts-ticket-v01"
        tus_ep="https://ymaqlcfjmdwncdbjprmw.storage.supabase.co/storage/v1/upload/resumable/sign"
        def signed_ticket(kind,index=None):
            payload={"kind":kind}
            if index is not None:payload["index"]=index
            r=S.post(ticket_ep,headers={"x-p586-token":TOKEN},json=payload,timeout=120);r.raise_for_status();return r.json()
        def b64s(v):return _b64.b64encode(v.encode()).decode()
        def tus_upload(local_path,kind,index=None,content_type="application/octet-stream"):
            info=signed_ticket(kind,index);nbytes=os.path.getsize(local_path)
            metadata=",".join([f"bucketName {b64s('bridge')}",f"objectName {b64s(info['path'])}",f"contentType {b64s(content_type)}",f"cacheControl {b64s('3600')}"])
            common={"Tus-Resumable":"1.0.0","x-signature":info["token"],"x-upsert":"true"}
            r=S.post(tus_ep,headers={**common,"Upload-Length":str(nbytes),"Upload-Metadata":metadata},data=b"",timeout=180);r.raise_for_status();loc=r.headers.get("Location") or r.headers.get("location")
            if not loc:raise RuntimeError("signed TUS returned no Location")
            if not loc.startswith("http"):loc="https://ymaqlcfjmdwncdbjprmw.storage.supabase.co"+loc
            offset=int(r.headers.get("Upload-Offset","0"));chunk=6*1024*1024
            with open(local_path,"rb") as fh:
                fh.seek(offset)
                while offset<nbytes:
                    data=fh.read(min(chunk,nbytes-offset))
                    p=S.patch(loc,headers={**common,"Upload-Offset":str(offset),"Content-Type":"application/offset+octet-stream"},data=data,timeout=600);p.raise_for_status();nxt=int(p.headers.get("Upload-Offset",str(offset+len(data))))
                    if nxt!=offset+len(data):raise RuntimeError(f"TUS offset mismatch {offset}+{len(data)} != {nxt}")
                    offset=nxt
            head=S.head(info["publicUrl"],timeout=120);head.raise_for_status();remote=int(head.headers.get("content-length","-1"))
            if remote!=nbytes:raise RuntimeError(f"remote size mismatch for {info['path']}: {remote} != {nbytes}")
            return info
        # Compact core bundle: all protocols, reports, ledgers, manifests and blind sheets,
        # excluding only the thousands of individual corpus crop images.
        core=Path(td)/"P586_PLANT_MORPHOLOGY_HUMAN_READABLE_CORE.zip"
        with zipfile.ZipFile(out,"r") as src,zipfile.ZipFile(core,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as dst:
            for zi in src.infolist():
                if zi.filename.startswith("02_CORPORA/") and "/images/" in zi.filename:continue
                dst.writestr(zi.filename,src.read(zi.filename))
        with zipfile.ZipFile(core) as zc:
            core_bad=zc.testzip();core_entries=len(zc.infolist())
        if core_bad:raise RuntimeError(f"core ZIP corrupt member {core_bad}")
        core_size=core.stat().st_size
        if core_size>45*1024*1024:raise RuntimeError(f"core ZIP exceeds 45 MiB: {core_size}")
        core_sha=hashlib.sha256(core.read_bytes()).hexdigest();core_info=tus_upload(core,"core",content_type="application/zip")
        part_size=45*1024*1024;parts=[]
        with open(out,"rb") as full:
            idx=0
            while True:
                data=full.read(part_size)
                if not data:break
                part=Path(td)/f"P586_PLANT_MORPHOLOGY_COMPLETE.zip.part{idx:03d}";part.write_bytes(data)
                psha=hashlib.sha256(data).hexdigest();pinfo=tus_upload(part,"part",idx)
                parts.append({"index":idx,"name":part.name,"bytes":len(data),"sha256":psha,"url":pinfo["publicUrl"]})
                print(json.dumps({"event":"part_uploaded","index":idx,"bytes":len(data),"parts_so_far":len(parts)},sort_keys=True),flush=True)
                part.unlink();idx+=1
        manifest={"protocol_id":PROTOCOL,"format":"binary concatenation in ascending part index","full_archive":{"filename":"P586_PLANT_MORPHOLOGY_COMPLETE.zip","bytes":size,"entries":entries,"sha256":digest,"zip_test":"pass"},"core_archive":{"filename":core.name,"bytes":core_size,"entries":core_entries,"sha256":core_sha,"url":core_info["publicUrl"],"zip_test":"pass"},"parts":parts,"analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
        manifest_path=Path(td)/"P586_PLANT_MORPHOLOGY_COMPLETE.manifest.json";manifest_path.write_bytes(stable_json(manifest));manifest_info=tus_upload(manifest_path,"manifest",content_type="application/json")
        urls="\n".join([f"  '{p['url']}'," for p in parts]).rstrip(',')
        ps='''$ErrorActionPreference = "Stop"\n$Output = Join-Path $PWD "P586_PLANT_MORPHOLOGY_COMPLETE.zip"\n$Urls = @(\n'''+urls+'''\n)\n$stream = [System.IO.File]::Create($Output)\ntry {\n  for ($i=0; $i -lt $Urls.Count; $i++) {\n    $tmp = Join-Path $env:TEMP ("p586_part_{0:D3}" -f $i)\n    Write-Host ("Downloading part {0}/{1}" -f ($i+1), $Urls.Count)\n    Invoke-WebRequest -Uri $Urls[$i] -OutFile $tmp\n    $bytes = [System.IO.File]::ReadAllBytes($tmp)\n    $stream.Write($bytes, 0, $bytes.Length)\n    Remove-Item $tmp -Force\n  }\n} finally { $stream.Dispose() }\n$hash=(Get-FileHash -Algorithm SHA256 $Output).Hash.ToLower()\nWrite-Host "SHA-256: $hash"\nif ($hash -ne "'''+digest+'''" ) { throw "SHA-256 mismatch" }\nWrite-Host "Archive assembled and verified: $Output"\n'''
        sh='''#!/usr/bin/env bash\nset -euo pipefail\nout="P586_PLANT_MORPHOLOGY_COMPLETE.zip"\n: > "$out"\n'''+"\n".join([f"curl -fL '{p['url']}' >> \"$out\"" for p in parts])+'''\nactual=$(sha256sum "$out" | awk '{print $1}')\nexpected="'''+digest+'''"\nprintf 'SHA-256: %s\\n' "$actual"\n[ "$actual" = "$expected" ] || { echo 'SHA-256 mismatch' >&2; exit 1; }\necho "Archive assembled and verified: $out"\n'''
        readme=f'''# P586 Plant-Morphology Evidence Bundle\n\nThe compact core ZIP contains all protocols, amendments, reports, manifests, CSV ledgers and blind evidence sheets. The full archive additionally contains every retained ordinary, masked, above-ground and reproductive crop.\n\nFull archive SHA-256: `{digest}`  \nFull archive bytes: `{size}`  \nFull archive entries: `{entries}`  \nParts: `{len(parts)}` in ascending `part000`, `part001`, ... order.\n\nUse `ASSEMBLE_P586_ARCHIVE.ps1` on Windows or `assemble_p586_archive.sh` on Linux/macOS. Both scripts download, concatenate and verify the archive.\n'''
        assets=[]
        for kind,name,data,ctype in [("powershell","ASSEMBLE_P586_ARCHIVE.ps1",ps.encode(),"text/plain"),("bash","assemble_p586_archive.sh",sh.encode(),"text/x-shellscript"),("readme","README.md",readme.encode(),"text/markdown")]:
            p=Path(td)/name;p.write_bytes(data);info=tus_upload(p,kind,content_type=ctype);assets.append({"kind":kind,"name":name,"url":info["publicUrl"],"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)})
        meta={"protocol_id":PROTOCOL,"delivery":"persistent core ZIP plus checksum-verified multipart full archive","core_url":core_info["publicUrl"],"manifest_url":manifest_info["publicUrl"],"assets":assets,"parts":len(parts),"full_archive_bytes":size,"full_archive_entries":entries,"full_archive_sha256":digest,"core_bytes":core_size,"core_entries":core_entries,"core_sha256":core_sha,"zip_test":"pass","analysis_result_sha256":analysis.get("result_sha256"),"blind_result_sha256":blind.get("result_sha256"),"closeout_result_sha256":close.get("closeout_result_sha256")}
'''
if source.count(old)!=1:raise RuntimeError(f"multipart patch point mismatch: {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
