#!/usr/bin/env python3
"""Non-inferential smoke test for the temporary frozen mask endpoint."""
from __future__ import annotations
import base64,hashlib,io,json,re,requests
from PIL import Image
ROOT_SOURCE="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/7f5edcb48b78c5a5b3f4f39dc10a7e4da2dfe922/experiments/palatino586_root_v01/p586_recover_localisation.py"
BRIDGE="https://ymaqlcfjmdwncdbjprmw.supabase.co/storage/v1/object/public/bridge/"
ENDPOINT="https://plant-mask-painter.lovable.app/api/p586-mask-v01"
EXPECTED_TOKEN_SHA="116954ffdc209c006292b4c5dcc96cbd0eddeb3e1c7ea788aeabb68b6855a929"

def main():
    status=requests.get(ENDPOINT,timeout=60); status.raise_for_status()
    root=requests.get(ROOT_SOURCE,timeout=60).text
    token=re.search(r'RUN_ID\s*=\s*"([^"]+)"',root).group(1)
    if hashlib.sha256(token.encode()).hexdigest()!=EXPECTED_TOKEN_SHA:
        raise RuntimeError("frozen token hash mismatch")
    manifest=requests.get(BRIDGE+"p586_plant_v01/target/whole_manifest_frozen.json",timeout=120).json()
    plant=next(x for x in manifest["plants"] if x["qa_status"]=="accept")
    source_url=BRIDGE+plant["crop_path"]
    source=requests.get(source_url,timeout=120).content
    response=requests.post(ENDPOINT,headers={"x-p586-token":token},json={"imageUrl":source_url},timeout=360)
    response.raise_for_status(); result=response.json(); output=base64.b64decode(result["pngBase64"])
    image=Image.open(io.BytesIO(output)); image.verify()
    report={
        "status":status.json(),
        "plant_id":plant["plant_id"],
        "source_sha256_local":hashlib.sha256(source).hexdigest(),
        "source_sha256_endpoint":result.get("sourceSha256"),
        "source_hash_matches":hashlib.sha256(source).hexdigest()==result.get("sourceSha256"),
        "model":result.get("model"),
        "prompt_sha256":result.get("promptSha256"),
        "response_id_present":bool(result.get("responseId")),
        "output_bytes":len(output),
        "output_sha256":hashlib.sha256(output).hexdigest(),
        "output_size":list(image.size),
        "output_format":image.format,
    }
    print("RESULT_JSON="+json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
