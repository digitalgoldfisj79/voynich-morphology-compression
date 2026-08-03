#!/usr/bin/env python3
"""Audited launcher: endpoint-name correction plus preregistered Amendment 02 control."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/26915172edf7fa04c633ed2126ff6b265d28fe11/experiments/palatino586_plant_v01/p586_embed_analyse.py"
source=requests.get(URL,timeout=120).text
source=source.replace(';UPLOAD=SUPA+"/functions/v1/p586-plant-upload-v01"',';UPLOAD_EP=SUPA+"/functions/v1/p586-plant-upload-v01"')
source=source.replace('S.post(UPLOAD,headers=', 'S.post(UPLOAD_EP,headers=')
old='"herb_d91d01bd5276","bsb1784"]'
new='"herb_d91d01bd5276","bsb1784","bnf_gr_2179"]'
if source.count(old)!=1:
    raise RuntimeError(f"control-panel patch expected once, found {source.count(old)}")
source=source.replace(old,new,1)
if 'UPLOAD_EP=SUPA+' not in source or 'S.post(UPLOAD_EP,headers=' not in source or '"bnf_gr_2179"' not in source:
    raise RuntimeError('audited analysis launcher corrections did not apply')
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
