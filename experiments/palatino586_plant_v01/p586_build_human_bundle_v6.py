#!/usr/bin/env python3
"""Run final bundle builder with edge-safe 5 MiB TUS chunks."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/8ab553a27ded407a8126f676c6b7d7ffc3691ee1/experiments/palatino586_plant_v01/p586_build_human_bundle_v5.py"
source=requests.get(URL,timeout=120).text
old='chunk=6*1024*1024'
new='chunk=5*1024*1024'
if source.count(old)!=1:raise RuntimeError(f"chunk patch mismatch: {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
