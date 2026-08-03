#!/usr/bin/env python3
"""Audited blind-review launcher adding the Amendment 02 control pool."""
import requests
URL="https://raw.githubusercontent.com/digitalgoldfisj79/voynich-morphology-compression/08da1b8c643379c7ce5786f61faaf1c2911c446d/experiments/palatino586_plant_v01/p586_blind_adjudication.py"
source=requests.get(URL,timeout=120).text
old='"herb_d91d01bd5276","bsb1784"]'
new='"herb_d91d01bd5276","bsb1784","bnf_gr_2179"]'
if source.count(old)!=1: raise RuntimeError(f"control patch expected once, found {source.count(old)}")
source=source.replace(old,new,1)
exec(compile(source,URL,"exec"),{"__name__":"__main__"})
