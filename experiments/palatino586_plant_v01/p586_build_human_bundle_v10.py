#!/usr/bin/env python3
"""Run the token-gated temporary artifact tunnel with valid import ordering."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/b867a20d803c506281f0a5f27be6d037cfa76e53/experiments/palatino586_plant_v01/p586_build_human_bundle_v9.py"
source=requests.get(URL,timeout=120).text
old="source='import http.server,threading,subprocess,secrets,urllib.parse\\n'+source"
new="source=source.replace('from __future__ import annotations\\n','from __future__ import annotations\\nimport http.server,threading,subprocess,secrets,urllib.parse\\n',1)"
if source.count(old)!=1:raise RuntimeError(f"import-order patch mismatch: {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
