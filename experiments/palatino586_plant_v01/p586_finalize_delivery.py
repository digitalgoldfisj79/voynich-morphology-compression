#!/usr/bin/env python3
"""Finalize persistent P586 bundle delivery from the uploaded manifest."""
from __future__ import annotations
import base64,hashlib,json,os,re,tempfile,requests
from pathlib import Path
SUPA="https://ymaqlcfjmdwncdbjprmw.supabase.co"
BRIDGE=SUPA+"/storage/v1/object/public/bridge/"
TICKET=SUPA+"/functions/v1/p586-bundle-parts-ticket-v01"
UPLOAD=SUPA+"/functions/v1/p586-plant-upload-v01"
TUS="https://ymaqlcfjmdwncdbjprmw.storage.supabase.co/storage/v1/upload/resumable/sign"
MANIFEST_PATH="p586_plant_v01/bundles/full/P586_PLANT_MORPHOLOGY_COMPLETE.manifest.json"
S=requests.Session();S.headers["User-Agent"]="P586DeliveryFinalize/0.1"
def get(u,timeout=180):r=S.get(u,timeout=timeout);r.raise_for_status();return r
def token():
 u="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py"
 return re.search(r'RUN_ID\s*=\s*"([^"]+)"',get(u).text).group(1)
TOKEN=token()
def b64s(v):return base64.b64encode(v.encode()).decode()
def stable(x):return json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False).encode()
def ticket(kind):
 r=S.post(TICKET,headers={"x-p586-token":TOKEN},json={"kind":kind},timeout=120);r.raise_for_status();return r.json()
def upload_file(path:Path,kind:str,content_type:str):
 info=ticket(kind);size=path.stat().st_size
 metadata=",".join([f"bucketName {b64s('bridge')}",f"objectName {b64s(info['path'])}",f"contentType {b64s(content_type)}",f"cacheControl {b64s('3600')}"])
 common={"Tus-Resumable":"1.0.0","x-signature":info["token"],"x-upsert":"true"}
 r=S.post(TUS,headers={**common,"Upload-Length":str(size),"Upload-Metadata":metadata},data=b"",timeout=180);r.raise_for_status();loc=r.headers.get("Location") or r.headers.get("location")
 if not loc:raise RuntimeError("signed TUS returned no Location")
 if not loc.startswith("http"):loc="https://ymaqlcfjmdwncdbjprmw.storage.supabase.co"+loc
 offset=int(r.headers.get("Upload-Offset","0"));chunk=6*1024*1024
 with open(path,"rb") as f:
  f.seek(offset)
  while offset<size:
   data=f.read(min(chunk,size-offset));p=S.patch(loc,headers={**common,"Upload-Offset":str(offset),"Content-Type":"application/offset+octet-stream"},data=data,timeout=600);p.raise_for_status();offset=int(p.headers.get("Upload-Offset",str(offset+len(data))))
 local=path.read_bytes();remote=get(info["publicUrl"]).content
 if remote!=local:raise RuntimeError(f"remote byte mismatch: {info['path']}")
 return {"kind":kind,"name":path.name,"url":info["publicUrl"],"bytes":size,"sha256":hashlib.sha256(local).hexdigest()}
def main():
 manifest_raw=get(BRIDGE+MANIFEST_PATH).content;manifest=json.loads(manifest_raw);parts=manifest["parts"];full=manifest["full_archive"];core=manifest["core_archive"]
 if len(parts)!=14 or sum(p["bytes"] for p in parts)!=full["bytes"]:raise RuntimeError("manifest part accounting failed")
 nl="\n";sq="'"
 urls=nl.join(["  "+sq+p["url"]+sq+"," for p in parts]).rstrip(",")
 ps=nl.join(['$ErrorActionPreference = "Stop"','$Output = Join-Path $PWD "P586_PLANT_MORPHOLOGY_COMPLETE.zip"','$Urls = @(',urls,')','$stream = [System.IO.File]::Create($Output)','try {','  for ($i=0; $i -lt $Urls.Count; $i++) {','    $tmp = Join-Path $env:TEMP ("p586_part_{0:D3}" -f $i)','    Write-Host ("Downloading part {0}/{1}" -f ($i+1), $Urls.Count)','    Invoke-WebRequest -Uri $Urls[$i] -OutFile $tmp','    $bytes = [System.IO.File]::ReadAllBytes($tmp)','    $stream.Write($bytes, 0, $bytes.Length)','    Remove-Item $tmp -Force','  }','} finally { $stream.Dispose() }','$hash=(Get-FileHash -Algorithm SHA256 $Output).Hash.ToLower()','Write-Host "SHA-256: $hash"',f'if ($hash -ne "{full["sha256"]}") {{ throw "SHA-256 mismatch" }}','Write-Host "Archive assembled and verified: $Output"',''])
 sh_lines=['#!/usr/bin/env bash','set -euo pipefail','out="P586_PLANT_MORPHOLOGY_COMPLETE.zip"',': > "$out"'];sh_lines += [f"curl -fL '{p['url']}' >> \"$out\"" for p in parts];sh_lines += ['actual=$(sha256sum "$out" | awk \'{print $1}\')',f'expected="{full["sha256"]}"','echo "SHA-256: $actual"','[ "$actual" = "$expected" ] || { echo \'SHA-256 mismatch\' >&2; exit 1; }','echo "Archive assembled and verified: $out"',''];sh=nl.join(sh_lines)
 readme=nl.join(['# P586 Plant-Morphology Evidence Bundle','',f'[Download the compact human-readable core ZIP]({core["url"]})','',f'[Download the full archive manifest]({BRIDGE+MANIFEST_PATH})','',f'Full archive SHA-256: `{full["sha256"]}`  ',f'Full archive bytes: `{full["bytes"]}`  ',f'Full archive entries: `{full["entries"]}`  ',f'Parts: `{len(parts)}`.','', 'Use `ASSEMBLE_P586_ARCHIVE.ps1` on Windows or `assemble_p586_archive.sh` on Linux/macOS. Each script downloads the fourteen parts, concatenates them in order, and verifies the complete archive hash.',''])
 with tempfile.TemporaryDirectory(prefix="p586_delivery_") as td:
  td=Path(td);assets=[]
  for kind,name,data,ctype in [("powershell","ASSEMBLE_P586_ARCHIVE.ps1",ps.encode(),"text/plain"),("bash","assemble_p586_archive.sh",sh.encode(),"text/x-shellscript"),("readme","README.md",readme.encode(),"text/markdown")]:
   p=td/name;p.write_bytes(data);assets.append(upload_file(p,kind,ctype))
 meta={"protocol_id":manifest["protocol_id"],"delivery":"persistent compact core ZIP plus checksum-verified multipart complete archive","core_archive":core,"full_archive":full,"manifest_url":BRIDGE+MANIFEST_PATH,"manifest_sha256":hashlib.sha256(manifest_raw).hexdigest(),"parts":parts,"assets":assets,"analysis_result_sha256":manifest["analysis_result_sha256"],"blind_result_sha256":manifest["blind_result_sha256"],"closeout_result_sha256":manifest["closeout_result_sha256"]}
 payload={"path":"p586_plant_v01/results/human_bundle.json","content_type":"application/json","data_b64":base64.b64encode(stable(meta)).decode()};r=S.post(UPLOAD,headers={"x-upload-token":TOKEN},json=payload,timeout=180);r.raise_for_status()
 print("RESULT_JSON="+json.dumps(meta,sort_keys=True),flush=True)
if __name__=="__main__":main()
