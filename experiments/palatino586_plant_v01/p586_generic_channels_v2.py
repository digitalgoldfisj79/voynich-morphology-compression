#!/usr/bin/env python3
"""Audited correction: run SAM2 in float32 so processor tensors and model weights agree.

No mask policy, threshold, proposal, QA or downstream analysis rule is changed.
"""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/588c7204dd3345cde56c51a98174494c8a766f90/experiments/palatino586_plant_v01/p586_generic_channels.py"
source=requests.get(URL,timeout=120).text
old='torch_dtype=torch.bfloat16 if dev=="cuda" else torch.float32'
new='torch_dtype=torch.float32'
if source.count(old)!=1:
    raise RuntimeError(f"expected exactly one SAM2 dtype expression, found {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
