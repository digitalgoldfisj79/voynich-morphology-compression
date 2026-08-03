#!/usr/bin/env python3
"""Audited launcher applying the endpoint-name correction to the frozen analysis runner."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/26915172edf7fa04c633ed2126ff6b265d28fe11/experiments/palatino586_plant_v01/p586_embed_analyse.py"
source=requests.get(URL,timeout=120).text
source=source.replace(';UPLOAD=SUPA+"/functions/v1/p586-plant-upload-v01"',';UPLOAD_EP=SUPA+"/functions/v1/p586-plant-upload-v01"')
source=source.replace('S.post(UPLOAD,headers=', 'S.post(UPLOAD_EP,headers=')
if 'UPLOAD_EP=SUPA+' not in source or 'S.post(UPLOAD_EP,headers=' not in source:
    raise RuntimeError('endpoint-name correction did not apply exactly')
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
