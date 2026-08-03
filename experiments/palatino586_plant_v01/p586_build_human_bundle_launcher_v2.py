#!/usr/bin/env python3
"""Audited bundle launcher adding the Amendment 02 corpus."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/b218fdfc6b30158d560194e8a7dd68456f65f374/experiments/palatino586_plant_v01/p586_build_human_bundle.py"
source=requests.get(URL,timeout=120).text
old='"herb_7ce7efc90e6d","herb_fafef9a26da5","herb_205bfb89efbc"]'
new='"herb_7ce7efc90e6d","herb_fafef9a26da5","herb_205bfb89efbc","bnf_gr_2179"]'
if source.count(old)!=1: raise RuntimeError(f"bundle corpus patch expected once, found {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
